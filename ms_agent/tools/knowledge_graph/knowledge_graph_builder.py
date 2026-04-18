import json
import os
import re
import requests
from typing import List, Dict, Any, Optional
from ms_agent.utils import get_logger
from .neo4j_connection import get_neo4j_connection
from .config import config

logger = get_logger()

_embedding_model = None
_model_loaded = False


class EmbeddingClient:
    """Embedding client that uses SiliconFlow API."""
    
    def __init__(self, api_key: str, model: str = "BAAI/bge-m3"):
        self.api_key = api_key
        self.model = model
        self.base_url = config.siliconflow_api_url
    
    def encode(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using SiliconFlow API."""
        try:
            embeddings_url = f"{self.base_url.rstrip('/')}/embeddings"
            response = requests.post(
                embeddings_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": self.model,
                    "input": text,
                    "encoding_format": "float"
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

_embedding_client = None


def _get_embedding_model():
    """Lazy load embedding model."""
    global _embedding_client
    if _embedding_client is None:
        if config.embedding_provider == 'siliconflow' and config.siliconflow_api_key:
            logger.info(f"Using SiliconFlow API with model: {config.embedding_model}")
            _embedding_client = EmbeddingClient(
                api_key=config.siliconflow_api_key,
                model=config.embedding_model
            )
        else:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Using local model: {config.embedding_model}")
                _embedding_client = SentenceTransformer(config.embedding_model)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                logger.warning("Knowledge graph will work without embeddings")
                _embedding_client = None
    return _embedding_client


class KnowledgeGraphBuilder:
    """
    Knowledge graph builder that constructs a graph from research evidence.
    """

    def __init__(self):
        """
        Initialize knowledge graph builder.
        """
        self.neo4j = get_neo4j_connection()

    def build_from_evidence(self, evidence_dir: str) -> None:
        """
        Build knowledge graph from evidence directory.
        
        Args:
            evidence_dir: Directory containing evidence files
        """
        logger.info(f"Building knowledge graph from evidence in {evidence_dir}")
        
        # Load evidence index
        index_path = os.path.join(evidence_dir, 'index.json')
        if not os.path.exists(index_path):
            logger.error(f"Evidence index not found at {index_path}")
            return
        
        with open(index_path, 'r', encoding='utf-8') as f:
            evidence_index = json.load(f)
        
        # Process each note in the index
        if 'notes' in evidence_index:
            for note_id, note_info in evidence_index['notes'].items():
                try:
                    self._process_evidence(note_id, note_info, evidence_dir)
                except Exception as e:
                    logger.error(f"Failed to process note {note_id}: {e}")

    def _process_evidence(self, evidence_id: str, evidence_info: Dict[str, Any], evidence_dir: str) -> None:
        """
        Process a single evidence item and add to knowledge graph.
        
        Args:
            evidence_id: Evidence ID
            evidence_info: Evidence information
            evidence_dir: Evidence directory
        """
        # Create document node
        document_id = f"doc_{evidence_id}"
        content = self._load_evidence_content(evidence_id, evidence_dir)
        
        if content:
            # Generate embedding
            embedding = self._generate_embedding(content)
            
            # Create document node
            query = """
                MERGE (d:Document {id: $id})
                SET d.title = $title,
                    d.url = $url,
                    d.content = $content,
                    d.timestamp = datetime()
                """
            # Get URL from sources if available
            url = ''
            sources = evidence_info.get('sources', [])
            if sources:
                url = sources[0].get('url', '')
            
            params = {
                "id": document_id,
                "title": evidence_info.get('title', ''),
                "url": url,
                "content": content,
            }
            
            # Only add embedding if available
            if embedding is not None:
                query = query.replace("d.timestamp = datetime()", "d.embedding = $embedding, d.timestamp = datetime()")
                # Check if embedding is already a list (from SiliconFlow API) or numpy array
                if hasattr(embedding, 'tolist'):
                    params["embedding"] = embedding.tolist()
                else:
                    params["embedding"] = embedding
            
            self.neo4j.run_query(query, params)
            
            # Extract entities and relationships
            entities = self._extract_entities(content)
            relationships = self._extract_relationships(content, entities)
            
            # Add entities to graph
            for entity in entities:
                self._add_entity(entity, document_id)
            
            # Add relationships to graph
            for rel in relationships:
                self._add_relationship(rel, document_id)

    def _load_evidence_content(self, evidence_id: str, evidence_dir: str) -> Optional[str]:
        """
        Load evidence content from file.
        
        Args:
            evidence_id: Evidence ID
            evidence_dir: Evidence directory
            
        Returns:
            Evidence content
        """
        notes_dir = os.path.join(evidence_dir, 'notes')
        evidence_file = os.path.join(notes_dir, f"note_{evidence_id}.md")
        
        if os.path.exists(evidence_file):
            with open(evidence_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def _generate_embedding(self, text: str) -> Any:
        """
        Generate embedding for text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        model = _get_embedding_model()
        if model is None:
            return None
        # Truncate text to avoid exceeding model limits
        text = text[:10000]
        return model.encode(text)

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using improved rules.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entities
        """
        entities = []
        
        # 文本预处理：清理Markdown格式和行尾换行符
        text = self._preprocess_text(text)
        
        # Define patterns to exclude (formatting, common stop words, etc.)
        exclude_patterns = [
            r'^\*\*.*\*\*$',  # Markdown bold formatting
            r'^##.*$',  # Markdown headers
            r'^-.*$',  # Markdown lists
            r'^\d{4}-\d{2}-\d{2}.*$',  # Dates
            r'^https?://.*$',  # URLs
            r'^`.*`$',  # Code/inline code
            r'^\[.*\]$',  # Markdown links
            r'^Note ID.*$',  # Metadata fields
            r'^Task ID.*$',  # Metadata fields
            r'^Quality Score.*$',  # Metadata fields
            r'^Created.*$',  # Metadata fields
            r'^Published.*$',  # Metadata fields
            r'^Source.*$',  # Metadata fields
            r'^Tags.*$',  # Metadata fields
            r'^Summary.*$',  # Common section headers
            r'^Content.*$',  # Common section headers
            r'^References.*$',  # Common section headers
            r'^Introduction.*$',  # Common section headers
            r'^Conclusion.*$',  # Common section headers
            r'^Key.*$',  # Generic terms
            r'^Primary.*$',  # Generic terms
            r'^Secondary.*$',  # Generic terms
            r'^Additional.*$',  # Generic terms
            r'^Metadata.*$',  # Section headers
            r'^Sources.*$',  # Section headers
            r'^- \*\*Note ID\*\*.*$',  # Metadata lines in notes
            r'^- \*\*Task ID\*\*.*$',  # Metadata lines in notes
            r'^- \*\*Tags\*\*.*$',  # Metadata lines in notes
            r'^- \*\*Quality Score\*\*.*$',  # Metadata lines in notes
            r'^- \*\*Created\*\*.*$',  # Metadata lines in notes
            r'^- \*\*Published\*\*.*$',  # Metadata lines in notes
            r'^- \*\*Source\*\*.*$',  # Metadata lines in notes
            r'^- \*\*Summary\*\*.*$',  # Metadata lines in notes
            r'^Quality$',  # Standalone metadata field names
            r'^Score$',
            r'^Task$',
            r'^Note$',
            r'^Tags$',
            r'^Created$',
            r'^Published$',
            r'^Source$',
            r'^Summary$',
            r'^Content$',
            r'^References$',
            r'^Key Features$',
            r'^Primary$',
            r'^Secondary$',
            r'^Additional$',
            r'^Furthermore.*$',  # Transition words
            r'^Moreover.*$',  # Transition words
            r'^Additionally.*$',  # Transition words
            r'^However.*$',  # Transition words
            r'^Therefore.*$',  # Transition words
            r'^Consequently.*$',  # Transition words
            r'^Thus.*$',  # Transition words
            r'^Hence.*$',  # Transition words
            r'^Overall.*$',  # Transition words
            r'^In.*$',  # Prepositions
            r'^On.*$',  # Prepositions
            r'^At.*$',  # Prepositions
            r'^By.*$',  # Prepositions
            r'^For.*$',  # Prepositions
            r'^With.*$',  # Prepositions
            r'^From.*$',  # Prepositions
            r'^To.*$',  # Prepositions
            r'^Of.*$',  # Prepositions
            r'^About.*$',  # Prepositions
            r'^Between.*$',  # Prepositions
            r'^Among.*$',  # Prepositions
            r'^Within.*$',  # Prepositions
            r'^Without.*$',  # Prepositions
            r'^Through.*$',  # Prepositions
            r'^During.*$',  # Prepositions
            r'^Since.*$',  # Prepositions
            r'^Until.*$',  # Prepositions
            r'^While.*$',  # Prepositions
            r'^Although.*$',  # Conjunctions
            r'^Though.*$',  # Conjunctions
            r'^Unless.*$',  # Conjunctions
            r'^Whether.*$',  # Conjunctions
            r'^Either.*$',  # Conjunctions
            r'^Neither.*$',  # Conjunctions
            r'^Both.*$',  # Conjunctions
            r'^And.*$',  # Conjunctions
            r'^Or.*$',  # Conjunctions
            r'^But.*$',  # Conjunctions
            r'^So.*$',  # Conjunctions
            r'^Yet.*$',  # Conjunctions
            r'^Nor.*$',  # Conjunctions
            r'^The.*$',  # Articles
            r'^A.*$',  # Articles
            r'^An.*$',  # Articles
            r'^This.*$',  # Demonstratives
            r'^That.*$',  # Demonstratives
            r'^These.*$',  # Demonstratives
            r'^Those.*$',  # Demonstratives
            r'^It.*$',  # Pronouns
            r'^Its.*$',  # Pronouns
            r'^They.*$',  # Pronouns
            r'^Their.*$',  # Pronouns
            r'^Them.*$',  # Pronouns
            r'^He.*$',  # Pronouns
            r'^She.*$',  # Pronouns
            r'^Him.*$',  # Pronouns
            r'^Her.*$',  # Pronouns
            r'^His.*$',  # Pronouns
            r'^We.*$',  # Pronouns
            r'^Us.*$',  # Pronouns
            r'^Our.*$',  # Pronouns
            r'^You.*$',  # Pronouns
            r'^Your.*$',  # Pronouns
            r'^Which.*$',  # Relative pronouns
            r'^Who.*$',  # Relative pronouns
            r'^Whom.*$',  # Relative pronouns
            r'^Whose.*$',  # Relative pronouns
            r'^Where.*$',  # Relative adverbs
            r'^When.*$',  # Relative adverbs
            r'^Why.*$',  # Relative adverbs
            r'^How.*$',  # Relative adverbs
            r'^What.*$',  # Interrogative pronouns
            r'^Such.*$',  # Determiners
            r'^Same.*$',  # Determiners
            r'^Other.*$',  # Determiners
            r'^Another.*$',  # Determiners
            r'^Each.*$',  # Determiners
            r'^Every.*$',  # Determiners
            r'^All.*$',  # Determiners
            r'^Some.*$',  # Determiners
            r'^Any.*$',  # Determiners
            r'^No.*$',  # Determiners
            r'^None.*$',  # Determiners
            r'^Both.*$',  # Determiners
            r'^Either.*$',  # Determiners
            r'^Neither.*$',  # Determiners
            r'^One.*$',  # Numbers
            r'^Two.*$',  # Numbers
            r'^Three.*$',  # Numbers
            r'^Four.*$',  # Numbers
            r'^Five.*$',  # Numbers
            r'^Six.*$',  # Numbers
            r'^Seven.*$',  # Numbers
            r'^Eight.*$',  # Numbers
            r'^Nine.*$',  # Numbers
            r'^Ten.*$',  # Numbers
            r'^First.*$',  # Ordinals
            r'^Second.*$',  # Ordinals
            r'^Third.*$',  # Ordinals
            r'^Last.*$',  # Ordinals
            r'^Next.*$',  # Ordinals
            r'^Previous.*$',  # Ordinals
            r'^Following.*$',  # Ordinals
            r'^Above.*$',  # Spatial
            r'^Below.*$',  # Spatial
            r'^Under.*$',  # Spatial
            r'^Over.*$',  # Spatial
            r'^Across.*$',  # Spatial
            r'^Through.*$',  # Spatial
            r'^Around.*$',  # Spatial
            r'^Between.*$',  # Spatial
            r'^Among.*$',  # Spatial
            r'^Within.*$',  # Spatial
            r'^Without.*$',  # Spatial
            r'^Beyond.*$',  # Spatial
            r'^Behind.*$',  # Spatial
            r'^Before.*$',  # Temporal
            r'^After.*$',  # Temporal
            r'^During.*$',  # Temporal
            r'^While.*$',  # Temporal
            r'^Since.*$',  # Temporal
            r'^Until.*$',  # Temporal
            r'^As.*$',  # Temporal
            r'^Once.*$',  # Temporal
            r'^Now.*$',  # Temporal
            r'^Then.*$',  # Temporal
            r'^Later.*$',  # Temporal
            r'^Soon.*$',  # Temporal
            r'^Already.*$',  # Temporal
            r'^Still.*$',  # Temporal
            r'^Yet.*$',  # Temporal
            r'^Again.*$',  # Temporal
            r'^Once.*$',  # Temporal
            r'^Twice.*$',  # Temporal
            r'^Never.*$',  # Temporal
            r'^Always.*$',  # Temporal
            r'^Often.*$',  # Temporal
            r'^Usually.*$',  # Temporal
            r'^Sometimes.*$',  # Temporal
            r'^Rarely.*$',  # Temporal
            r'^Seldom.*$',  # Temporal
            r'^Frequently.*$',  # Temporal
            r'^Generally.*$',  # Temporal
            r'^Typically.*$',  # Temporal
            r'^Commonly.*$',  # Temporal
            r'^Normally.*$',  # Temporal
            r'^Usually.*$',  # Temporal
            r'^Regularly.*$',  # Temporal
            r'^Continuously.*$',  # Temporal
            r'^Constantly.*$',  # Temporal
            r'^Permanently.*$',  # Temporal
            r'^Temporarily.*$',  # Temporal
            r'^Briefly.*$',  # Temporal
            r'^Quickly.*$',  # Temporal
            r'^Slowly.*$',  # Temporal
            r'^Rapidly.*$',  # Temporal
            r'^Gradually.*$',  # Temporal
            r'^Suddenly.*$',  # Temporal
            r'^Immediately.*$',  # Temporal
            r'^Instantly.*$',  # Temporal
            r'^Directly.*$',  # Temporal
            r'^Indirectly.*$',  # Temporal
            r'^Clearly.*$',  # Temporal
            r'^Obviously.*$',  # Temporal
            r'^Certainly.*$',  # Temporal
            r'^Definitely.*$',  # Temporal
            r'^Absolutely.*$',  # Temporal
            r'^Completely.*$',  # Temporal
            r'^Totally.*$',  # Temporal
            r'^Fully.*$',  # Temporal
            r'^Partially.*$',  # Temporal
            r'^Partly.*$',  # Temporal
            r'^Mainly.*$',  # Temporal
            r'^Primarily.*$',  # Temporal
            r'^Mostly.*$',  # Temporal
            r'^Largely.*$',  # Temporal
            r'^Significantly.*$',  # Temporal
            r'^Considerably.*$',  # Temporal
            r'^Substantially.*$',  # Temporal
            r'^Greatly.*$',  # Temporal
            r'^Highly.*$',  # Temporal
            r'^Extremely.*$',  # Temporal
            r'^Very.*$',  # Temporal
            r'^Quite.*$',  # Temporal
            r'^Rather.*$',  # Temporal
            r'^Fairly.*$',  # Temporal
            r'^Pretty.*$',  # Temporal
            r'^Somewhat.*$',  # Temporal
            r'^Slightly.*$',  # Temporal
            r'^Little.*$',  # Temporal
            r'^Much.*$',  # Temporal
            r'^More.*$',  # Temporal
            r'^Less.*$',  # Temporal
            r'^Most.*$',  # Temporal
            r'^Least.*$',  # Temporal
            r'^Best.*$',  # Temporal
            r'^Better.*$',  # Temporal
            r'^Worse.*$',  # Temporal
            r'^Worst.*$',  # Temporal
            r'^Good.*$',  # Temporal
            r'^Bad.*$',  # Temporal
            r'^High.*$',  # Temporal
            r'^Low.*$',  # Temporal
            r'^Big.*$',  # Temporal
            r'^Small.*$',  # Temporal
            r'^Large.*$',  # Temporal
            r'^Long.*$',  # Temporal
            r'^Short.*$',  # Temporal
            r'^Fast.*$',  # Temporal
            r'^Slow.*$',  # Temporal
            r'^New.*$',  # Temporal
            r'^Old.*$',  # Temporal
            r'^Young.*$',  # Temporal
            r'^Recent.*$',  # Temporal
            r'^Modern.*$',  # Temporal
            r'^Current.*$',  # Temporal
            r'^Present.*$',  # Temporal
            r'^Past.*$',  # Temporal
            r'^Future.*$',  # Temporal
            r'^Possible.*$',  # Temporal
            r'^Impossible.*$',  # Temporal
            r'^Likely.*$',  # Temporal
            r'^Unlikely.*$',  # Temporal
            r'^Probable.*$',  # Temporal
            r'^Improbable.*$',  # Temporal
            r'^Certain.*$',  # Temporal
            r'^Uncertain.*$',  # Temporal
            r'^Sure.*$',  # Temporal
            r'^Unsure.*$',  # Temporal
            r'^True.*$',  # Temporal
            r'^False.*$',  # Temporal
            r'^Real.*$',  # Temporal
            r'^Actual.*$',  # Temporal
            r'^Virtual.*$',  # Temporal
            r'^Theoretical.*$',  # Temporal
            r'^Practical.*$',  # Temporal
            r'^Useful.*$',  # Temporal
            r'^Useless.*$',  # Temporal
            r'^Important.*$',  # Temporal
            r'^Unimportant.*$',  # Temporal
            r'^Significant.*$',  # Temporal
            r'^Insignificant.*$',  # Temporal
            r'^Major.*$',  # Temporal
            r'^Minor.*$',  # Temporal
            r'^Main.*$',  # Temporal
            r'^Primary.*$',  # Temporal
            r'^Secondary.*$',  # Temporal
            r'^Essential.*$',  # Temporal
            r'^Non-essential.*$',  # Temporal
            r'^Necessary.*$',  # Temporal
            r'^Unnecessary.*$',  # Temporal
            r'^Required.*$',  # Temporal
            r'^Optional.*$',  # Temporal
            r'^Available.*$',  # Temporal
            r'^Unavailable.*$',  # Temporal
            r'^Accessible.*$',  # Temporal
            r'^Inaccessible.*$',  # Temporal
            r'^Visible.*$',  # Temporal
            r'^Invisible.*$',  # Temporal
            r'^Clear.*$',  # Temporal
            r'^Unclear.*$',  # Temporal
            r'^Obvious.*$',  # Temporal
            r'^Not.*$',  # Negation
            r'^No.*$',  # Negation
            r'^None.*$',  # Negation
            r'^Never.*$',  # Negation
            r'^Neither.*$',  # Negation
            r'^Nor.*$',  # Negation
            r'^Cannot.*$',  # Negation
            r'^Can.*$',  # Modal
            r'^Could.*$',  # Modal
            r'^Would.*$',  # Modal
            r'^Should.*$',  # Modal
            r'^Will.*$',  # Modal
            r'^Shall.*$',  # Modal
            r'^May.*$',  # Modal
            r'^Might.*$',  # Modal
            r'^Must.*$',  # Modal
            r'^Ought.*$',  # Modal
            r'^Need.*$',  # Modal
            r'^Dare.*$',  # Modal
            r'^Used.*$',  # Passive
            r'^Made.*$',  # Passive
            r'^Done.*$',  # Passive
            r'^Seen.*$',  # Passive
            r'^Found.*$',  # Passive
            r'^Given.*$',  # Passive
            r'^Taken.*$',  # Passive
            r'^Had.*$',  # Passive
            r'^Got.*$',  # Passive
            r'^Became.*$',  # Passive
            r'^Came.*$',  # Passive
            r'^Went.*$',  # Passive
            r'^Said.*$',  # Passive
            r'^Told.*$',  # Passive
            r'^Asked.*$',  # Passive
            r'^Answered.*$',  # Passive
            r'^Called.*$',  # Passive
            r'^Named.*$',  # Passive
            r'^Known.*$',  # Passive
            r'^Thought.*$',  # Passive
            r'^Believed.*$',  # Passive
            r'^Considered.*$',  # Passive
            r'^Regarded.*$',  # Passive
            r'^Treated.*$',  # Passive
            r'^Handled.*$',  # Passive
            r'^Managed.*$',  # Passive
            r'^Controlled.*$',  # Passive
            r'^Directed.*$',  # Passive
            r'^Led.*$',  # Passive
            r'^Guided.*$',  # Passive
            r'^Helped.*$',  # Passive
            r'^Supported.*$',  # Passive
            r'^Assisted.*$',  # Passive
            r'^Served.*$',  # Passive
            r'^Worked.*$',  # Passive
            r'^Played.*$',  # Passive
            r'^Acted.*$',  # Passive
            r'^Performed.*$',  # Passive
            r'^Executed.*$',  # Passive
            r'^Implemented.*$',  # Passive
            r'^Applied.*$',  # Passive
            r'^Utilized.*$',  # Passive
            r'^Employed.*$',  # Passive
            r'^Used.*$',  # Passive
            r'^Included.*$',  # Passive
            r'^Contained.*$',  # Passive
            r'^Consisted.*$',  # Passive
            r'^Comprised.*$',  # Passive
            r'^Involved.*$',  # Passive
            r'^Required.*$',  # Passive
            r'^Needed.*$',  # Passive
            r'^Demanded.*$',  # Passive
            r'^Expected.*$',  # Passive
            r'^Anticipated.*$',  # Passive
            r'^Predicted.*$',  # Passive
            r'^Estimated.*$',  # Passive
            r'^Calculated.*$',  # Passive
            r'^Measured.*$',  # Passive
            r'^Analyzed.*$',  # Passive
            r'^Examined.*$',  # Passive
            r'^Investigated.*$',  # Passive
            r'^Studied.*$',  # Passive
            r'^Researched.*$',  # Passive
            r'^Explored.*$',  # Passive
            r'^Discovered.*$',  # Passive
            r'^Found.*$',  # Passive
            r'^Identified.*$',  # Passive
            r'^Recognized.*$',  # Passive
            r'^Acknowledged.*$',  # Passive
            r'^Accepted.*$',  # Passive
            r'^Approved.*$',  # Passive
            r'^Authorized.*$',  # Passive
            r'^Permitted.*$',  # Passive
            r'^Allowed.*$',  # Passive
            r'^Enabled.*$',  # Passive
            r'^Facilitated.*$',  # Passive
            r'^Supported.*$',  # Passive
            r'^Encouraged.*$',  # Passive
            r'^Promoted.*$',  # Passive
            r'^Developed.*$',  # Passive
            r'^Created.*$',  # Passive
            r'^Produced.*$',  # Passive
            r'^Generated.*$',  # Passive
            r'^Built.*$',  # Passive
            r'^Constructed.*$',  # Passive
            r'^Established.*$',  # Passive
            r'^Founded.*$',  # Passive
            r'^Started.*$',  # Passive
            r'^Began.*$',  # Passive
            r'^Launched.*$',  # Passive
            r'^Initiated.*$',  # Passive
            r'^Introduced.*$',  # Passive
            r'^Presented.*$',  # Passive
            r'^Showed.*$',  # Passive
            r'^Displayed.*$',  # Passive
            r'^Demonstrated.*$',  # Passive
            r'^Illustrated.*$',  # Passive
            r'^Explained.*$',  # Passive
            r'^Described.*$',  # Passive
            r'^Discussed.*$',  # Passive
            r'^Reviewed.*$',  # Passive
            r'^Summarized.*$',  # Passive
            r'^Concluded.*$',  # Passive
            r'^Finished.*$',  # Passive
            r'^Completed.*$',  # Passive
            r'^Ended.*$',  # Passive
            r'^Stopped.*$',  # Passive
            r'^Ceased.*$',  # Passive
            r'^Terminated.*$',  # Passive
            r'^Closed.*$',  # Passive
            r'^Opened.*$',  # Passive
            r'^Started.*$',  # Passive
            r'^Began.*$',  # Passive
            r'^Commenced.*$',  # Passive
        ]
        
        # Compile patterns for efficiency
        exclude_regex = re.compile('|'.join(exclude_patterns), re.IGNORECASE)
        
        # Split text into sentences for better context
        sentences = re.split(r'[.!?]+', text)
        
        # Extract potential entities using improved patterns
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Pattern 1: Capitalized words (proper nouns)
            capitalized_pattern = r'\b[A-Z][a-zA-Z]{2,}\b'
            capitalized_matches = re.findall(capitalized_pattern, sentence)
            
            # Pattern 2: Technical terms with hyphens or underscores
            technical_pattern = r'\b[a-zA-Z]+[-_][a-zA-Z]+\b'
            technical_matches = re.findall(technical_pattern, sentence)
            
            # Pattern 3: Acronyms (all caps, 2-4 letters)
            acronym_pattern = r'\b[A-Z]{2,4}\b'
            acronym_matches = re.findall(acronym_pattern, sentence)
            
            # Pattern 4: Multi-word phrases (proper nouns with 2-4 words)
            # 改进的短语提取模式，确保能够提取完整的实体
            phrase_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b'
            phrase_matches = re.findall(phrase_pattern, sentence)
            
            # Pattern 5: 带连字符的多词短语
            hyphenated_phrase_pattern = r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+\b'
            hyphenated_phrase_matches = re.findall(hyphenated_phrase_pattern, sentence)
            
            # Combine all potential entities
            potential_entities = set(capitalized_matches + technical_matches + acronym_matches + phrase_matches + hyphenated_phrase_matches)
            
            for entity_name in potential_entities:
                # 实体归一化
                normalized_entity = self._normalize_entity(entity_name)
                
                # Skip if matches exclusion patterns
                if exclude_regex.match(normalized_entity):
                    continue
                
                # Skip if too short or too long
                if len(normalized_entity) < 3 or len(normalized_entity) > 50:
                    continue
                
                # Skip if it's a common word (case-insensitive)
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their', 'them', 'he', 'she', 'him', 'her', 'his', 'we', 'us', 'our', 'you', 'your', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'what', 'such', 'same', 'other', 'another', 'each', 'every', 'all', 'some', 'any', 'no', 'none', 'both', 'either', 'neither'}
                if normalized_entity.lower() in common_words:
                    continue
                
                # Skip if it's a number
                if normalized_entity.isdigit():
                    continue
                
                # Skip if it contains only special characters
                if not re.search(r'[a-zA-Z]', normalized_entity):
                    continue
                
                # Find context for the entity
                entity_idx = sentence.find(entity_name)
                if entity_idx == -1:
                    continue
                
                context_start = max(0, entity_idx - 50)
                context_end = min(len(sentence), entity_idx + len(entity_name) + 50)
                context = sentence[context_start:context_end]
                
                # Determine entity type based on patterns
                entity_type = self._infer_entity_type(normalized_entity, context)
                
                entities.append({
                    "id": f"entity_{hash(normalized_entity)}",
                    "name": normalized_entity,
                    "type": entity_type,
                    "context": context
                })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_key = (entity['name'], entity['type'])
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        # 实体合并：合并相似或相同的实体
        merged_entities = self._merge_similar_entities(unique_entities)
        
        return self._filter_low_quality_entities(merged_entities)
        
    def _merge_similar_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并相似或相同的实体"""
        if not entities:
            return []
        
        # 按实体名称分组
        entity_groups = {}
        for entity in entities:
            name = entity['name'].lower()
            if name not in entity_groups:
                entity_groups[name] = []
            entity_groups[name].append(entity)
        
        # 合并每组中的实体
        merged = []
        for name, group in entity_groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # 选择质量最高的实体作为代表
                best_entity = max(group, key=lambda e: self._calculate_quality_score(e['name'], e.get('context', '')))
                merged.append(best_entity)
        
        return merged
        
    def _preprocess_text(self, text: str) -> str:
        """预处理文本，清理Markdown格式和行尾换行符"""
        # 清理行尾换行符
        text = re.sub(r'\n+', ' ', text)
        # 清理多余的空格
        text = re.sub(r'\s+', ' ', text)
        # 清理Markdown格式的粗体
        text = re.sub(r'\*\*', '', text)
        return text
        
    def _normalize_entity(self, entity_name: str) -> str:
        """归一化实体，处理大小写不一致问题"""
        # 对于多词短语，保持首字母大写，其余小写
        words = entity_name.split()
        normalized_words = []
        for word in words:
            # 对于缩写词，保持全大写
            if word.isupper() and len(word) > 1:
                normalized_words.append(word)
            else:
                normalized_words.append(word.capitalize())
        return ' '.join(normalized_words)

    def _infer_entity_type(self, entity_name: str, context: str) -> str:
        """Infer entity type based on context."""
        type_indicators = {
            'Organization': ['company', 'corp', 'inc', 'founded', 'labs', 'tech', 'enterprise', 'firm', 'business', 'startup', 'organization', 'institute', 'university', 'college', 'school', 'government', 'agency', 'department'],
            'Technology': ['framework', 'model', 'system', 'platform', 'architecture', 'engine', 'protocol', 'algorithm', 'network', 'technology', 'tool', 'software', 'hardware', 'device', 'equipment'],
            'Product': ['product', 'launch', 'release', 'version', 'tool', 'package', 'library', 'sdk', 'api', 'assistant', 'application', 'app', 'service', 'solution', 'system'],
            'Method': ['algorithm', 'technique', 'approach', 'method', 'strategy', 'optimization', 'training', 'inference', 'methodology', 'procedure', 'process', 'technique', 'practice'],
            'Person': ['researcher', 'scientist', 'engineer', 'developer', 'author', 'founder', 'ceo', 'cto', 'director', 'person', 'individual', 'human', 'expert', 'specialist', 'professional'],
            'Dataset': ['dataset', 'corpus', 'benchmark', 'evaluation', 'training data', 'test data', 'data set', 'database', 'collection', 'repository'],
            'Metric': ['accuracy', 'precision', 'recall', 'f1', 'score', 'loss', 'perplexity', 'bleu', 'rouge', 'metric', 'measure', 'indicator', 'statistic', 'score'],
            'Concept': ['concept', 'idea', 'theory', 'principle', 'notion', 'thought', 'belief', 'hypothesis', 'proposition'],
            'Process': ['process', 'procedure', 'method', 'technique', 'approach', 'system', 'mechanism', 'operation', 'function'],
            'Event': ['event', 'conference', 'meeting', 'workshop', 'symposium', 'convention', 'gathering', 'occurrence'],
            'Theory': ['theory', 'model', 'framework', 'paradigm', 'concept', 'hypothesis', 'principle', 'law'],
            'Field': ['field', 'domain', 'area', 'discipline', 'subject', 'topic', 'sector', 'industry'],
            'Task': ['task', 'job', 'assignment', 'project', 'work', 'activity', 'function', 'operation'],
            'Challenge': ['challenge', 'problem', 'issue', 'difficulty', 'obstacle', 'barrier', 'hurdle', 'complication'],
            'Opportunity': ['opportunity', 'chance', 'possibility', 'prospect', 'potential', 'opening', 'window'],
            'Solution': ['solution', 'answer', 'response', 'resolution', 'fix', 'remedy', 'cure', 'treatment'],
        }
        
        # 基于实体名称的模式匹配
        if re.match(r'^[A-Z]{2,4}$', entity_name):
            return "Acronym"
        if re.search(r'[-_]', entity_name):
            return "TechnicalTerm"
        if re.match(r'^[0-9]+(\.[0-9]+)?$', entity_name):
            return "Number"
        if re.match(r'^https?://', entity_name):
            return "URL"
        
        # 基于上下文的类型推断
        context_lower = context.lower()
        for etype, indicators in type_indicators.items():
            for indicator in indicators:
                if indicator in context_lower:
                    return etype
        
        # 基于实体名称的语义分析
        if entity_name.endswith('tion') or entity_name.endswith('sion') or entity_name.endswith('ment'):
            return "Process"
        if entity_name.endswith('er') or entity_name.endswith('or') or entity_name.endswith('ist'):
            return "Person"
        if entity_name.endswith('ity') or entity_name.endswith('ness') or entity_name.endswith('ship'):
            return "Concept"
        
        # 基于大小写模式的推断
        if entity_name.istitle() and len(entity_name) > 2:
            # 检查是否是多词短语
            if ' ' in entity_name:
                return "ProperNoun"
            else:
                return "Entity"
        
        return "Entity"

    def _filter_low_quality_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out low-quality entities."""
        metadata_words = {'quality', 'score', 'task', 'note', 'tags', 'created', 'published', 'source', 'summary', 'content', 'references', 'metadata', 'primary', 'secondary', 'additional', 'key', 'features'}
        
        quality_words = {
            'research', 'paper', 'study', 'work', 'system', 'method', 'approach', 'technique',
            'model', 'framework', 'problem', 'solution', 'result', 'finding', 'conclusion',
            'analysis', 'task', 'issue', 'point', 'view', 'case', 'use', 'used', 'using',
            'based', 'number', 'part', 'kind', 'type', 'way', 'form', 'set', 'group',
            'class', 'level', 'stage', 'field', 'area', 'domain', 'section', 'note',
            'term', 'fact', 'idea', 'concept', 'thought', 'example', 'instance',
            'question', 'answer', 'data', 'information', 'knowledge', 'understanding',
            'experience', 'practice', 'application', 'development', 'improvement',
            'performance', 'quality', 'score', 'value', 'cost', 'benefit', 'advantage',
            'disadvantage', 'challenge', 'opportunity', 'strategy', 'process', 'step',
            'feature', 'aspect', 'component', 'element', 'factor', 'condition', 'situation',
            'state', 'status', 'phase', 'activity', 'action', 'operation', 'function',
            'purpose', 'goal', 'objective', 'aim', 'target', 'limit', 'range', 'scope',
            'scale', 'size', 'length', 'width', 'height', 'depth', 'weight', 'speed',
            'rate', 'frequency', 'percent', 'percentage', 'ratio', 'proportion', 'amount',
            'quantity', 'number', 'figure', 'outcome', 'output', 'input', 'resource',
            'source', 'material', 'tool', 'technology', 'algorithm', 'protocol', 'standard',
            'requirement', 'specification', 'definition', 'description', 'explanation',
            'interpretation', 'evaluation', 'assessment', 'comparison', 'contrast',
            'difference', 'similarity', 'relation', 'connection', 'association',
            'correlation', 'dependence', 'interaction', 'relationship', 'think', 'deep',
            'world', 'super', 'enhanced', 'providing', 'google', 'microsoft', 'nvidia',
        }
        
        # 无意义的实体组合模式
        meaningless_patterns = [
            r'Content\s+The',  # 无意义的组合
            r'Summary\s+Multi',  # 无意义的组合
            r'Note\s+ID',  # 元数据字段
            r'Task\s+ID',  # 元数据字段
            r'Quality\s+Score',  # 元数据字段
            r'Created\s+On',  # 元数据字段
            r'Published\s+On',  # 元数据字段
            r'Source\s+URL',  # 元数据字段
            r'Tags\s+List',  # 元数据字段
        ]
        
        filtered = []
        for entity in entities:
            name = entity['name']
            name_lower = name.lower()
            
            # 基本过滤
            if len(name) < 3:
                continue
            if name_lower in {'the', 'a', 'an', 'and', 'or', 'but'}:
                continue
            if name.isdigit():
                continue
            if not re.search(r'[a-zA-Z]', name):
                continue
            if re.match(r'^[a-z]+$', name_lower):
                continue
            if name_lower in metadata_words:
                continue
            if name_lower in quality_words:
                continue
            if name.startswith('`') or name.endswith('`'):
                continue
            if re.match(r'^\d+$', name):
                continue
            
            # 无意义实体组合过滤
            is_meaningless = False
            for pattern in meaningless_patterns:
                if re.search(pattern, name, re.IGNORECASE):
                    is_meaningless = True
                    break
            if is_meaningless:
                continue
            
            # 质量评分
            quality_score = self._calculate_quality_score(name, entity.get('context', ''))
            if quality_score < 0.3:  # 质量阈值
                continue
            
            filtered.append(entity)
        
        return filtered
        
    def _calculate_quality_score(self, entity_name: str, context: str) -> float:
        """计算实体质量评分"""
        score = 0.0
        
        # 长度评分
        length = len(entity_name)
        if 3 <= length <= 50:
            score += 0.2
        elif length > 50:
            score += 0.1
        
        # 词数评分
        word_count = len(entity_name.split())
        if 1 <= word_count <= 4:
            score += 0.2
        
        # 大写字母评分（专有名词）
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', entity_name):
            score += 0.2
        
        # 技术术语评分
        if re.search(r'[-_]', entity_name):
            score += 0.2
        
        # 上下文相关性评分
        if context:
            # 检查实体在上下文中的使用情况
            if entity_name in context and len(context) > 20:
                score += 0.2
        
        # 语义评分（简单的关键词匹配）
        meaningful_keywords = {'algorithm', 'model', 'system', 'method', 'technique', 'framework', 'platform', 'architecture', 'protocol', 'network', 'dataset', 'metric', 'company', 'product', 'person', 'organization'}
        for keyword in meaningful_keywords:
            if keyword in entity_name.lower():
                score += 0.1
                break
        
        return min(score, 1.0)

    def _extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using improved rules.
        
        Args:
            text: Text to extract relationships from
            entities: List of entities
            
        Returns:
            List of relationships
        """
        relationships = []
        
        # Define relationship indicators (verbs and phrases that suggest relationships)
        relationship_indicators = {
            'causes': ['causes', 'caused', 'leads to', 'led to', 'results in', 'resulted in', 'produces', 'produced', 'creates', 'created', 'generates', 'generated'],
            'enables': ['enables', 'enabled', 'allows', 'allowed', 'permits', 'permitted', 'facilitates', 'facilitated', 'supports', 'supported'],
            'requires': ['requires', 'required', 'needs', 'needed', 'depends on', 'depended on', 'relies on', 'relied on'],
            'contains': ['contains', 'contained', 'includes', 'included', 'consists of', 'consisted of', 'comprises', 'comprised', 'involves', 'involved'],
            'improves': ['improves', 'improved', 'enhances', 'enhanced', 'increases', 'increased', 'boosts', 'boosted', 'optimizes', 'optimized'],
            'reduces': ['reduces', 'reduced', 'decreases', 'decreased', 'minimizes', 'minimized', 'lowers', 'lowered', 'cuts', 'cut'],
            'prevents': ['prevents', 'prevented', 'avoids', 'avoided', 'stops', 'stopped', 'blocks', 'blocked', 'inhibits', 'inhibited'],
            'similar_to': ['similar to', 'similar as', 'like', 'resembles', 'resembled', 'analogous to', 'comparable to'],
            'different_from': ['different from', 'differs from', 'unlike', 'distinct from', 'separate from'],
            'part_of': ['part of', 'portion of', 'component of', 'element of', 'aspect of', 'feature of'],
            'uses': ['uses', 'used', 'utilizes', 'utilized', 'employs', 'employed', 'applies', 'applied', 'implements', 'implemented'],
            'applied_to': ['applied to', 'used in', 'utilized in', 'employed in', 'implemented in'],
            'based_on': ['based on', 'built on', 'founded on', 'grounded in', 'rooted in', 'derived from'],
            'affects': ['affects', 'affected', 'impacts', 'impacted', 'influences', 'influenced', 'shapes', 'shaped'],
            'related_to': ['related to', 'associated with', 'connected to', 'linked to', 'tied to'],
            'example_of': ['example of', 'instance of', 'case of', 'illustration of'],
            'defined_as': ['defined as', 'means', 'refers to', 'stands for', 'represents'],
            'located_in': ['located in', 'found in', 'situated in', 'positioned in', 'placed in'],
            'occurs_in': ['occurs in', 'happens in', 'takes place in', 'appears in'],
            'belongs_to': ['belongs to', 'owned by', 'possessed by', 'held by'],
            'follows': ['follows', 'followed', 'comes after', 'came after', 'succeeds', 'succeeded'],
            'precedes': ['precedes', 'preceded', 'comes before', 'came before', 'precedes', 'preceeded'],
        }
        
        # Split text into sentences for better context
        sentences = re.split(r'[.!?]+', text)
        
        # Create entity lookup for faster access
        entity_dict = {entity['name']: entity for entity in entities}
        
        # 关系去重集合
        seen_relationships = set()
        
        # Process each sentence
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue
            
            # Find entities present in this sentence
            sentence_entities = []
            for entity in entities:
                if entity['name'] in sentence:
                    sentence_entities.append(entity)
            
            # Skip if less than 2 entities in sentence
            if len(sentence_entities) < 2:
                continue
            
            # Check for relationship indicators
            found_relationship = False
            for rel_type, indicators in relationship_indicators.items():
                for indicator in indicators:
                    if indicator in sentence.lower():
                        # Find entities around the indicator
                        indicator_idx = sentence.lower().find(indicator)
                        
                        # Find entities before and after the indicator
                        before_entities = []
                        after_entities = []
                        
                        for entity in sentence_entities:
                            entity_idx = sentence.find(entity['name'])
                            if entity_idx < indicator_idx:
                                before_entities.append(entity)
                            else:
                                after_entities.append(entity)
                        
                        # Create relationships between entities before and after the indicator
                        if before_entities and after_entities:
                            for source_entity in before_entities:
                                for target_entity in after_entities:
                                    # Skip if same entity
                                    if source_entity['name'] == target_entity['name']:
                                        continue
                                    
                                    # 生成关系键，用于去重
                                    # 为了避免方向问题，对源和目标进行排序
                                    sorted_entities = sorted([source_entity['name'], target_entity['name']])
                                    rel_key = (sorted_entities[0], sorted_entities[1], rel_type)
                                    
                                    # 检查关系是否已存在
                                    if rel_key in seen_relationships:
                                        continue
                                    seen_relationships.add(rel_key)
                                    
                                    # 计算关系质量评分
                                    rel_quality = self._calculate_relationship_quality(source_entity, target_entity, sentence, indicator)
                                    if rel_quality < 0.3:  # 关系质量阈值
                                        continue
                                    
                                    # Extract context around the relationship
                                    start_idx = max(0, min(sentence.find(source_entity['name']), sentence.find(target_entity['name'])) - 30)
                                    end_idx = min(len(sentence), max(sentence.find(source_entity['name']) + len(source_entity['name']), 
                                                                                   sentence.find(target_entity['name']) + len(target_entity['name'])) + 30)
                                    context = sentence[start_idx:end_idx]
                                    
                                    relationships.append({
                                        "source": source_entity['id'],
                                        "target": target_entity['id'],
                                        "type": rel_type.upper(),
                                        "context": context,
                                        "key": rel_key
                                    })
        
                                    found_relationship = True
            
            # If no specific relationship indicator found, create generic RELATED_TO relationships
            # but only if entities are close to each other in the sentence
            if not found_relationship:
                for i, entity1 in enumerate(sentence_entities):
                    for j, entity2 in enumerate(sentence_entities):
                        if i < j:  # Avoid duplicate relationships
                            # Calculate distance between entities in the sentence
                            idx1 = sentence.find(entity1['name'])
                            idx2 = sentence.find(entity2['name'])
                            if idx1 != -1 and idx2 != -1:
                                distance = abs(idx1 - idx2)
                                # Only create relationship if entities are close (within 50 characters)
                                if distance < 50:
                                    # 生成关系键，用于去重
                                    sorted_entities = sorted([entity1['name'], entity2['name']])
                                    rel_key = (sorted_entities[0], sorted_entities[1], 'related_to')
                                    
                                    # 检查关系是否已存在
                                    if rel_key not in seen_relationships:
                                        seen_relationships.add(rel_key)
                                        
                                        # 计算关系质量评分
                                        rel_quality = self._calculate_relationship_quality(entity1, entity2, sentence, 'related to')
                                        if rel_quality < 0.2:  # 关系质量阈值
                                            continue
                                        
                                        # Extract context around the relationship
                                        start_idx = max(0, min(idx1, idx2) - 30)
                                        end_idx = min(len(sentence), max(idx1 + len(entity1['name']), idx2 + len(entity2['name'])) + 30)
                                        context = sentence[start_idx:end_idx]
                                        
                                        relationships.append({
                                            "source": entity1['id'],
                                            "target": entity2['id'],
                                            "type": "RELATED_TO",
                                            "context": context,
                                            "key": rel_key
                                        })
        
        return relationships
        
    def _calculate_relationship_quality(self, source_entity: Dict[str, Any], target_entity: Dict[str, Any], sentence: str, indicator: str) -> float:
        """计算关系质量评分"""
        score = 0.0
        
        # 实体质量评分
        source_quality = self._calculate_quality_score(source_entity['name'], source_entity.get('context', ''))
        target_quality = self._calculate_quality_score(target_entity['name'], target_entity.get('context', ''))
        score += (source_quality + target_quality) / 2 * 0.3
        
        # 关系指示器强度
        indicator_strength = {
            'causes': 0.9, 'enables': 0.8, 'requires': 0.8, 'contains': 0.7,
            'improves': 0.7, 'reduces': 0.7, 'prevents': 0.7, 'similar_to': 0.6,
            'different_from': 0.6, 'part_of': 0.8, 'uses': 0.8, 'applied_to': 0.7,
            'based_on': 0.8, 'affects': 0.7, 'related_to': 0.5, 'example_of': 0.6,
            'defined_as': 0.9, 'located_in': 0.7, 'occurs_in': 0.6, 'belongs_to': 0.8,
            'follows': 0.6, 'precedes': 0.6
        }
        
        # 提取关系类型
        rel_type = None
        relationship_indicators = {
            'causes': ['causes', 'caused', 'leads to', 'led to', 'results in', 'resulted in', 'produces', 'produced', 'creates', 'created', 'generates', 'generated'],
            'enables': ['enables', 'enabled', 'allows', 'allowed', 'permits', 'permitted', 'facilitates', 'facilitated', 'supports', 'supported'],
            'requires': ['requires', 'required', 'needs', 'needed', 'depends on', 'depended on', 'relies on', 'relied on'],
            'contains': ['contains', 'contained', 'includes', 'included', 'consists of', 'consisted of', 'comprises', 'comprised', 'involves', 'involved'],
            'improves': ['improves', 'improved', 'enhances', 'enhanced', 'increases', 'increased', 'boosts', 'boosted', 'optimizes', 'optimized'],
            'reduces': ['reduces', 'reduced', 'decreases', 'decreased', 'minimizes', 'minimized', 'lowers', 'lowered', 'cuts', 'cut'],
            'prevents': ['prevents', 'prevented', 'avoids', 'avoided', 'stops', 'stopped', 'blocks', 'blocked', 'inhibits', 'inhibited'],
            'similar_to': ['similar to', 'similar as', 'like', 'resembles', 'resembled', 'analogous to', 'comparable to'],
            'different_from': ['different from', 'differs from', 'unlike', 'distinct from', 'separate from'],
            'part_of': ['part of', 'portion of', 'component of', 'element of', 'aspect of', 'feature of'],
            'uses': ['uses', 'used', 'utilizes', 'utilized', 'employs', 'employed', 'applies', 'applied', 'implements', 'implemented'],
            'applied_to': ['applied to', 'used in', 'utilized in', 'employed in', 'implemented in'],
            'based_on': ['based on', 'built on', 'founded on', 'grounded in', 'rooted in', 'derived from'],
            'affects': ['affects', 'affected', 'impacts', 'impacted', 'influences', 'influenced', 'shapes', 'shaped'],
            'related_to': ['related to', 'associated with', 'connected to', 'linked to', 'tied to'],
            'example_of': ['example of', 'instance of', 'case of', 'illustration of'],
            'defined_as': ['defined as', 'means', 'refers to', 'stands for', 'represents'],
            'located_in': ['located in', 'found in', 'situated in', 'positioned in', 'placed in'],
            'occurs_in': ['occurs in', 'happens in', 'takes place in', 'appears in'],
            'belongs_to': ['belongs to', 'owned by', 'possessed by', 'held by'],
            'follows': ['follows', 'followed', 'comes after', 'came after', 'succeeds', 'succeeded'],
            'precedes': ['precedes', 'preceded', 'comes before', 'came before', 'precedes', 'preceeded'],
        }
        
        for r_type, indicators in relationship_indicators.items():
            if indicator in indicators:
                rel_type = r_type
                break
        
        if rel_type and rel_type in indicator_strength:
            score += indicator_strength[rel_type] * 0.4
        else:
            score += 0.5 * 0.4
        
        # 实体在句子中的距离
        source_idx = sentence.find(source_entity['name'])
        target_idx = sentence.find(target_entity['name'])
        distance = abs(source_idx - target_idx)
        max_distance = len(sentence)
        distance_score = max(0, 1 - (distance / max_distance))
        score += distance_score * 0.3
        
        return min(score, 1.0)

    def _add_entity(self, entity: Dict[str, Any], document_id: str) -> None:
        """
        Add entity to knowledge graph.
        
        Args:
            entity: Entity information
            document_id: Document ID
        """
        self.neo4j.run_query(
            """
            MERGE (e:Entity {id: $id})
            SET e.name = $name,
                e.type = $type,
                e.context = $context
            MERGE (d:Document {id: $document_id})
            MERGE (d)-[:MENTIONS]->(e)
            """,
            {
                "id": entity['id'],
                "name": entity['name'],
                "type": entity['type'],
                "context": entity['context'],
                "document_id": document_id
            }
        )

    def _add_relationship(self, relationship: Dict[str, Any], document_id: str) -> None:
        """
        Add relationship to knowledge graph.
        
        Args:
            relationship: Relationship information
            document_id: Document ID
        """
        self.neo4j.run_query(
            """
            MERGE (s:Entity {id: $source})
            MERGE (t:Entity {id: $target})
            MERGE (s)-[r:RELATED_TO {context: $context}]->(t)
            MERGE (d:Document {id: $document_id})
            MERGE (d)-[:MENTIONS]->(s)
            MERGE (d)-[:MENTIONS]->(t)
            """,
            {
                "source": relationship['source'],
                "target": relationship['target'],
                "context": relationship['context'],
                "document_id": document_id
            }
        )

    def build_from_query(self, query: str) -> None:
        """
        Build knowledge graph from a research query.
        
        Args:
            query: Research query
        """
        # Create query node
        query_id = f"query_{hash(query)}"
        embedding = self._generate_embedding(query)
        
        self.neo4j.run_query(
            """
            MERGE (q:Query {id: $id})
            SET q.text = $text,
                q.embedding = $embedding,
                q.timestamp = datetime()
            """,
            {
                "id": query_id,
                "text": query,
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            }
        )

    def update_relationships(self) -> None:
        """
        Update relationships between entities based on co-occurrence.
        """
        # This is a simplified implementation
        # In production, use more sophisticated relationship extraction
        self.neo4j.run_query(
            """
            MATCH (e1:Entity)-[:MENTIONS]->(d:Document)<-[:MENTIONS]-(e2:Entity)
            WHERE e1 <> e2
            MERGE (e1)-[r:CO_OCCURS]->(e2)
            SET r.strength = coalesce(r.strength, 0) + 1
            """
        )

import json
import os
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
            response = requests.post(
                self.base_url,
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
        
        # Process each evidence item
        for evidence_id, evidence_info in evidence_index.items():
            try:
                self._process_evidence(evidence_id, evidence_info, evidence_dir)
            except Exception as e:
                logger.error(f"Failed to process evidence {evidence_id}: {e}")

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
            params = {
                "id": document_id,
                "title": evidence_info.get('title', ''),
                "url": evidence_info.get('url', ''),
                "content": content,
            }
            
            # Only add embedding if available
            if embedding is not None:
                query = query.replace("d.timestamp = datetime()", "d.embedding = $embedding, d.timestamp = datetime()")
                params["embedding"] = embedding.tolist()
            
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
        evidence_file = os.path.join(notes_dir, f"{evidence_id}.md")
        
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
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entities
        """
        # Simple entity extraction (can be replaced with more sophisticated NER)
        entities = []
        # Example: extract proper nouns and key terms
        # This is a simplified implementation
        # In production, use a proper NER model
        words = text.split()
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 2:
                entities.append({
                    "id": f"entity_{hash(word)}",
                    "name": word,
                    "type": "Entity",
                    "context": " ".join(words[max(0, i-2):min(len(words), i+3)])
                })
        return entities

    def _extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities.
        
        Args:
            text: Text to extract relationships from
            entities: List of entities
            
        Returns:
            List of relationships
        """
        # Simple relationship extraction
        relationships = []
        # This is a simplified implementation
        # In production, use a proper relation extraction model
        entity_names = [entity['name'] for entity in entities]
        
        for i, entity1 in enumerate(entity_names):
            for j, entity2 in enumerate(entity_names):
                if i < j and entity1 in text and entity2 in text:
                    # Find context between entities
                    entity1_idx = text.find(entity1)
                    entity2_idx = text.find(entity2)
                    start = min(entity1_idx, entity2_idx)
                    end = max(entity1_idx + len(entity1), entity2_idx + len(entity2))
                    context = text[max(0, start-50):min(len(text), end+50)]
                    
                    relationships.append({
                        "source": f"entity_{hash(entity1)}",
                        "target": f"entity_{hash(entity2)}",
                        "type": "RELATED_TO",
                        "context": context
                    })
        return relationships

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
            MERGE (d)-[:MENTIONS]->(r)
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
                "embedding": embedding.tolist()
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

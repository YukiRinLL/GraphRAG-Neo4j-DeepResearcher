from typing import List, Dict, Any, Optional
from ms_agent.utils import get_logger
from .neo4j_connection import get_neo4j_connection
from .config import config
from .knowledge_graph_builder import _get_embedding_model

logger = get_logger()


class KnowledgeGraphRetriever:
    """
    Knowledge graph retriever that searches for relevant information in the knowledge graph.
    """

    def __init__(self):
        """
        Initialize knowledge graph retriever.
        """
        self.neo4j = get_neo4j_connection()

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant information from knowledge graph based on query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant information
        """
        logger.info(f"Retrieving from knowledge graph for query: {query}")
        
        if top_k is None:
            top_k = config.top_k
        
        # Generate embedding for query
        embedding_model = _get_embedding_model()
        if embedding_model is None:
            logger.warning("No embedding model available, falling back to basic search")
            return self._basic_search(query, top_k)
        
        query_embedding = embedding_model.encode(query)
        
        # Search for relevant documents
        document_results = self._search_documents(query_embedding, top_k)
        
        # Search for relevant entities
        entity_results = self._search_entities(query_embedding, top_k)
        
        # Combine results
        results = []
        results.extend(document_results)
        results.extend(entity_results)
        
        # Sort by relevance
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results[:top_k]

    def _search_documents(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using vector similarity.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        # Use Neo4j's vector similarity search
        # Note: This requires Neo4j 5.12+ with vector index
        try:
            result = self.neo4j.run_query(
                """
                CALL db.index.vector.queryNodes('document_embedding', $top_k, $embedding)
                YIELD node, score
                RETURN node.id as id, node.title as title, node.url as url, node.content as content, score
                """,
                {
                    "top_k": top_k,
                    "embedding": query_embedding.tolist()
                }
            )
            
            documents = []
            for record in result:
                documents.append({
                    "type": "document",
                    "id": record['id'],
                    "title": record['title'],
                    "url": record['url'],
                    "content": record['content'],
                    "score": record['score']
                })
            return documents
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to keyword search: {e}")
            # Fallback to keyword search
            return self._keyword_search_documents(query_embedding, top_k)

    def _basic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Basic search when no embedding model is available.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant information
        """
        # Basic document search
        result = self.neo4j.run_query(
            """
            MATCH (d:Document)
            WHERE d.title CONTAINS $query OR d.content CONTAINS $query
            RETURN d.id as id, d.title as title, d.url as url, d.content as content
            LIMIT $top_k
            """,
            {"query": query, "top_k": top_k}
        )
        
        results = []
        for record in result:
            results.append({
                "type": "document",
                "id": record['id'],
                "title": record['title'],
                "url": record['url'],
                "content": record['content'],
                "score": 0.5  # Default score
            })
        
        # If no results, return all documents
        if not results:
            result = self.neo4j.run_query(
                """
                MATCH (d:Document)
                RETURN d.id as id, d.title as title, d.url as url, d.content as content
                LIMIT $top_k
                """,
                {"top_k": top_k}
            )
            
            for record in result:
                results.append({
                    "type": "document",
                    "id": record['id'],
                    "title": record['title'],
                    "url": record['url'],
                    "content": record['content'],
                    "score": 0.3  # Lower score for default results
                })
        
        return results

    def _keyword_search_documents(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Fallback keyword search for documents.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        # This is a simplified fallback
        # In production, use a proper keyword search
        result = self.neo4j.run_query(
            """
            MATCH (d:Document)
            RETURN d.id as id, d.title as title, d.url as url, d.content as content
            LIMIT $top_k
            """,
            {"top_k": top_k}
        )
        
        documents = []
        for record in result:
            documents.append({
                "type": "document",
                "id": record['id'],
                "title": record['title'],
                "url": record['url'],
                "content": record['content'],
                "score": 0.5  # Default score for fallback
            })
        return documents

    def _search_entities(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Search for relevant entities using vector similarity.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of relevant entities
        """
        # Use Neo4j's vector similarity search
        try:
            result = self.neo4j.run_query(
                """
                CALL db.index.vector.queryNodes('entity_embedding', $top_k, $embedding)
                YIELD node, score
                RETURN node.id as id, node.name as name, node.type as type, node.context as context, score
                """,
                {
                    "top_k": top_k,
                    "embedding": query_embedding.tolist()
                }
            )
            
            entities = []
            for record in result:
                entities.append({
                    "type": "entity",
                    "id": record['id'],
                    "name": record['name'],
                    "type": record['type'],
                    "context": record['context'],
                    "score": record['score']
                })
            return entities
        except Exception as e:
            logger.warning(f"Vector search for entities failed: {e}")
            return []

    def get_related_entities(self, entity_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """
        Get related entities for a given entity.
        
        Args:
            entity_id: Entity ID
            depth: Search depth
            
        Returns:
            List of related entities
        """
        result = self.neo4j.run_query(
            """
            MATCH (e:Entity {id: $entity_id})-[r*1..$depth]-(related:Entity)
            RETURN related.id as id, related.name as name, related.type as type, 
                   [rel in r | type(rel)] as relationships
            """,
            {
                "entity_id": entity_id,
                "depth": depth
            }
        )
        
        related_entities = []
        for record in result:
            related_entities.append({
                "id": record['id'],
                "name": record['name'],
                "type": record['type'],
                "relationships": record['relationships']
            })
        return related_entities

    def get_entity_connections(self, entity_name: str) -> Dict[str, Any]:
        """
        Get connections for a given entity name.
        
        Args:
            entity_name: Entity name
            
        Returns:
            Entity connections
        """
        result = self.neo4j.run_query(
            """
            MATCH (e:Entity {name: $entity_name})-[r]-(related)
            RETURN related.id as id, related.name as name, related.type as type, type(r) as relationship
            """,
            {"entity_name": entity_name}
        )
        
        connections = {
            "entity": entity_name,
            "related": []
        }
        
        for record in result:
            connections["related"].append({
                "id": record['id'],
                "name": record['name'],
                "type": record['type'],
                "relationship": record['relationship']
            })
        
        return connections

    def create_vector_indexes(self) -> None:
        """
        Create vector indexes for efficient search.
        """
        # Create vector index for documents
        try:
            self.neo4j.run_query(
                """
                CREATE VECTOR INDEX document_embedding IF NOT EXISTS
                FOR (d:Document)
                ON (d.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
            )
            logger.info("Created vector index for documents")
        except Exception as e:
            logger.warning(f"Failed to create document vector index: {e}")
        
        # Create vector index for entities
        try:
            self.neo4j.run_query(
                """
                CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (e:Entity)
                ON (e.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
            )
            logger.info("Created vector index for entities")
        except Exception as e:
            logger.warning(f"Failed to create entity vector index: {e}")

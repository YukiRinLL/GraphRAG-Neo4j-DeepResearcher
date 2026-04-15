import os
from typing import Optional, Dict, Any
from neo4j import GraphDatabase, Driver
from ms_agent.utils import get_logger
from .config import config

logger = get_logger()


class Neo4jConnection:
    """
    Neo4j connection manager that handles connection to Neo4j database.
    """

    def __init__(self, 
                 uri: Optional[str] = None, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password
        """
        self.uri = uri or config.neo4j_uri
        self.username = username or config.neo4j_username
        self.password = password or config.neo4j_password
        self.driver: Optional[Driver] = None

    def connect(self) -> None:
        """
        Connect to Neo4j database.
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j database at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def disconnect(self) -> None:
        """
        Disconnect from Neo4j database.
        """
        if self.driver:
            self.driver.close()
            logger.info("Disconnected from Neo4j database")

    def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Run a Cypher query against Neo4j database.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query result
        """
        if not self.driver:
            self.connect()
        
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return result

    def create_constraints(self) -> None:
        """
        Create necessary constraints for the knowledge graph.
        """
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self.run_query(constraint)
                logger.info(f"Created constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Failed to create constraint: {e}")

    def clear_database(self) -> None:
        """
        Clear all data from the Neo4j database.
        """
        try:
            self.run_query("MATCH (n) DETACH DELETE n")
            logger.info("Cleared Neo4j database")
        except Exception as e:
            logger.error(f"Failed to clear Neo4j database: {e}")
            raise


# Singleton instance
_neo4j_connection: Optional[Neo4jConnection] = None


def get_neo4j_connection() -> Neo4jConnection:
    """
    Get singleton Neo4j connection instance.
    
    Returns:
        Neo4jConnection instance
    """
    global _neo4j_connection
    if _neo4j_connection is None:
        _neo4j_connection = Neo4jConnection()
        _neo4j_connection.connect()
        _neo4j_connection.create_constraints()
    return _neo4j_connection


def close_neo4j_connection() -> None:
    """
    Close Neo4j connection.
    """
    global _neo4j_connection
    if _neo4j_connection:
        _neo4j_connection.disconnect()
        _neo4j_connection = None

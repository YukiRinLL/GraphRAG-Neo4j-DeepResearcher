import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ms_agent.tools.knowledge_graph.neo4j_connection import get_neo4j_connection
from ms_agent.utils import get_logger

logger = get_logger()


async def clear_neo4j_database():
    """
    Clear all data from Neo4j database.
    """
    try:
        neo4j = get_neo4j_connection()
        
        logger.info("Starting to clear Neo4j database...")
        
        # Delete all relationships and nodes
        neo4j.run_query("MATCH (n) DETACH DELETE n")
        
        logger.info("Successfully cleared Neo4j database")
        
        # Verify the database is empty
        result = neo4j.run_query("MATCH (n) RETURN count(n) as node_count")
        for record in result:
            node_count = record['node_count']
            logger.info(f"Database verification: {node_count} nodes remaining")
        
        return "Database cleared successfully"
        
    except Exception as e:
        logger.error(f"Failed to clear Neo4j database: {e}")
        return f"Failed to clear database: {str(e)}"


if __name__ == "__main__":
    result = asyncio.run(clear_neo4j_database())
    print(result)
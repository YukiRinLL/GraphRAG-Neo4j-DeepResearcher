import asyncio
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

async def test_neo4j_connection():
    """Test Neo4j connection."""
    print("Testing Neo4j Connection...")
    
    try:
        from neo4j import AsyncGraphDatabase
        
        uri = os.environ.get('NEO4J_URI', 'neo4j://localhost:7687')
        username = os.environ.get('NEO4J_USERNAME', 'neo4j')
        password = os.environ.get('NEO4J_PASSWORD', 'password')
        
        print(f"Connecting to Neo4j at {uri} with user {username}...")
        
        driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        
        async with driver.session() as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            print(f"Connection successful! Test result: {record['test']}")
            
            # Check database statistics
            result = await session.run("MATCH (n) RETURN count(n) as node_count")
            record = await result.single()
            print(f"Current node count in database: {record['node_count']}")
            
            result = await session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            record = await result.single()
            print(f"Current relationship count in database: {record['rel_count']}")
        
        await driver.close()
        print("\nNeo4j Connection Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"Error during Neo4j connection test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_neo4j_connection())
    sys.exit(0 if success else 1)
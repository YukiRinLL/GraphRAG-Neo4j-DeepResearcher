import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_neo4j_connection_variants():
    """Test Neo4j connection with different URI formats."""
    print("Testing Neo4j Connection with Different Formats...")
    
    # Test different URI formats
    uri_variants = [
        "neo4j://localhost:7687",
        "bolt://localhost:7687",
        "bolt+s://localhost:7687",
        "neo4j+s://localhost:7687"
    ]
    
    username = os.environ.get('NEO4J_USERNAME', 'neo4j')
    password = os.environ.get('NEO4J_PASSWORD', 'LUOCANYU')
    
    print(f"Username: {username}")
    print(f"Password: {password}")
    print()
    
    for uri in uri_variants:
        try:
            print(f"Testing URI: {uri}")
            from neo4j import AsyncGraphDatabase
            
            driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
            
            async with driver.session() as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                print(f"  Connection successful! Test result: {record['test']}")
                
                # Check database statistics
                result = await session.run("MATCH (n) RETURN count(n) as node_count")
                record = await result.single()
                print(f"  Current node count in database: {record['node_count']}")
                
                result = await session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                record = await result.single()
                print(f"  Current relationship count in database: {record['rel_count']}")
            
            await driver.close()
            print(f"  URI {uri} works!")
            print()
            return True
            
        except Exception as e:
            print(f"  URI {uri} failed: {e}")
            print()
    
    print("All URI formats failed.")
    return False

if __name__ == "__main__":
    success = asyncio.run(test_neo4j_connection_variants())
    sys.exit(0 if success else 1)
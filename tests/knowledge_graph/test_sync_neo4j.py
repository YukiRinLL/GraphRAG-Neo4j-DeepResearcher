import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=False)

from neo4j import GraphDatabase

uri = os.environ.get('NEO4J_URI', 'neo4j://localhost:7687')
username = os.environ.get('NEO4J_USERNAME', 'neo4j')
password = os.environ.get('NEO4J_PASSWORD', 'LUOCANYU')

print(f"Testing synchronous Neo4j connection...")
print(f"URI: {uri}")
print(f"Username: {username}")
print(f"Password: {password}")
print()

try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        record = result.single()
        print(f"Connection successful! Test result: {record['test']}")
        
        # Check database statistics
        result = session.run("MATCH (n) RETURN count(n) as node_count")
        record = result.single()
        print(f"Current node count in database: {record['node_count']}")
        
        result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
        record = result.single()
        print(f"Current relationship count in database: {record['rel_count']}")
    
    driver.close()
    print("\nSynchronous Neo4j Connection Test Completed Successfully!")
    
except Exception as e:
    print(f"Error during synchronous Neo4j connection test: {e}")
    import traceback
    traceback.print_exc()
import os
import sys
from dotenv import load_dotenv, find_dotenv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv(find_dotenv(), override=False)

print("Testing KnowledgeGraphBuilder...")
try:
    from ms_agent.tools.knowledge_graph.knowledge_graph_builder import KnowledgeGraphBuilder
    
    print("Initializing KnowledgeGraphBuilder...")
    builder = KnowledgeGraphBuilder()
    
    print("KnowledgeGraphBuilder initialized successfully!")
    
    # Test if we can access the Neo4j connection
    print("Testing Neo4j connection through builder...")
    result = builder.neo4j.run_query("RETURN 1 as test")
    record = result.single()
    print(f"Connection test result: {record['test']}")
    
    # Check database statistics
    print("Checking database statistics...")
    result = builder.neo4j.run_query("MATCH (n) RETURN count(n) as node_count")
    record = result.single()
    print(f"Current node count in database: {record['node_count']}")
    
    result = builder.neo4j.run_query("MATCH ()-[r]->() RETURN count(r) as rel_count")
    record = result.single()
    print(f"Current relationship count in database: {record['rel_count']}")
    
    print("\nKnowledgeGraphBuilder Test Completed Successfully!")
    
except Exception as e:
    print(f"Error during KnowledgeGraphBuilder test: {e}")
    import traceback
    traceback.print_exc()
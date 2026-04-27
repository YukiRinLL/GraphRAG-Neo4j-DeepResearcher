# Test script for knowledge graph functionality
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ms_agent.tools.knowledge_graph import get_neo4j_connection

# Test Neo4j connection
def test_neo4j_connection():
    print("Testing Neo4j connection...")
    try:
        neo4j = get_neo4j_connection()
        print("✅ Neo4j connection successful")
        # Test a simple query
        result = neo4j.run_query("RETURN 1")
        print(f"✅ Query executed successfully: {list(result)}")
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing knowledge graph functionality...")
    print("=" * 60)
    
    # Run tests
    test_neo4j_connection()
    
    print("=" * 60)
    print("Test completed!")

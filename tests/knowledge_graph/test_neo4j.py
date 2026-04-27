#!/usr/bin/env python3
# Simple test script for Neo4j connection

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
        # Consume the result properly
        records = list(result)
        print(f"✅ Query executed successfully: {records}")
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Neo4j connection...")
    print("=" * 60)
    
    test_neo4j_connection()
    
    print("=" * 60)
    print("Test completed!")

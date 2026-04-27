import asyncio
import os
import sys
from dotenv import load_dotenv, find_dotenv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv(find_dotenv(), override=False)

async def test_knowledge_graph():
    """Test knowledge graph functionality."""
    print("Testing Knowledge Graph Functionality...")
    
    try:
        from ms_agent.tools.knowledge_graph.knowledge_graph_tool import KnowledgeGraphTool
        
        print("Initializing Knowledge Graph Tool...")
        kg_tool = KnowledgeGraphTool()
        
        print("Knowledge Graph Tool initialized successfully!")
        
        # Test building graph from a sample directory
        print("Testing graph building from sample data...")
        sample_dir = "sample/data"
        if os.path.exists(sample_dir):
            result = await kg_tool.build_graph(sample_dir)
            print(f"Graph build result: {result}")
        else:
            print(f"Sample directory {sample_dir} not found, skipping graph build test.")
        
        print("\nKnowledge Graph Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"Error during knowledge graph test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_knowledge_graph())
    sys.exit(0 if success else 1)
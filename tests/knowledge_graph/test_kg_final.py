import os
import sys
import asyncio
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
        
        print("Initializing KnowledgeGraphTool...")
        kg_tool = KnowledgeGraphTool()
        
        print("KnowledgeGraphTool initialized successfully!")
        
        # Test building graph from sample data
        print("\nTesting graph building from sample data...")
        sample_dir = "sample/data"
        if os.path.exists(sample_dir):
            print(f"Found sample directory: {sample_dir}")
            result = await kg_tool.build_graph(sample_dir)
            print(f"Graph build result: {result}")
        else:
            print(f"Sample directory {sample_dir} not found.")
            print("Creating a simple test document...")
            
            # Create a test evidence directory structure
            test_evidence_dir = "output/test_evidence"
            os.makedirs(test_evidence_dir, exist_ok=True)
            os.makedirs(os.path.join(test_evidence_dir, "notes"), exist_ok=True)
            
            # Create a simple test document
            test_content = """
            Large Language Models (LLMs) have seen significant advances in recent years.
            Transformers architecture has revolutionized natural language processing.
            Attention mechanisms enable models to focus on relevant parts of input.
            Pre-training on large datasets followed by fine-tuning is a common approach.
            GPT-4 and Claude are examples of advanced LLMs.
            """
            
            # Create evidence file
            evidence_file = os.path.join(test_evidence_dir, "notes", "test_1.md")
            with open(evidence_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # Create index file
            index_file = os.path.join(test_evidence_dir, "index.json")
            import json
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "test_1": {
                        "title": "Test Document",
                        "url": "https://example.com/test",
                        "created_at": "2026-04-15"
                    }
                }, f)
            
            print(f"Created test evidence directory: {test_evidence_dir}")
            
            # Build graph from test data
            result = await kg_tool.build_graph(test_evidence_dir)
            print(f"Graph build result: {result}")
        
        print("\nKnowledge Graph Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"Error during knowledge graph test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_knowledge_graph())
    import sys
    sys.exit(0 if success else 1)
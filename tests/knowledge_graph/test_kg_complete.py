import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=False)

print("Testing Knowledge Graph Functionality...")
try:
    from ms_agent.tools.knowledge_graph.knowledge_graph_builder import KnowledgeGraphBuilder
    
    print("Initializing KnowledgeGraphBuilder...")
    builder = KnowledgeGraphBuilder()
    
    print("KnowledgeGraphBuilder initialized successfully!")
    
    # Check database statistics
    print("Checking database statistics...")
    result = builder.neo4j.run_query("MATCH (n) RETURN count(n) as node_count")
    record = result.single()
    node_count = record['node_count']
    print(f"Current node count in database: {node_count}")
    
    result = builder.neo4j.run_query("MATCH ()-[r]->() RETURN count(r) as rel_count")
    record = result.single()
    rel_count = record['rel_count']
    print(f"Current relationship count in database: {rel_count}")
    
    # Test building graph from sample data
    print("\nTesting graph building from sample data...")
    sample_dir = "sample/data"
    if os.path.exists(sample_dir):
        print(f"Found sample directory: {sample_dir}")
        result = await builder.build_graph(sample_dir)
        print(f"Graph build result: {result}")
        
        # Check statistics again after building
        print("\nChecking database statistics after build...")
        result = builder.neo4j.run_query("MATCH (n) RETURN count(n) as node_count")
        record = result.single()
        new_node_count = record['node_count']
        print(f"New node count in database: {new_node_count} (added {new_node_count - node_count})")
        
        result = builder.neo4j.run_query("MATCH ()-[r]->() RETURN count(r) as rel_count")
        record = result.single()
        new_rel_count = record['rel_count']
        print(f"New relationship count in database: {new_rel_count} (added {new_rel_count - rel_count})")
    else:
        print(f"Sample directory {sample_dir} not found.")
        print("Creating a simple test document...")
        
        # Create a simple test document
        test_content = """
        Large Language Models (LLMs) have seen significant advances in recent years.
        Transformers architecture has revolutionized natural language processing.
        Attention mechanisms enable models to focus on relevant parts of input.
        Pre-training on large datasets followed by fine-tuning is a common approach.
        """
        
        result = builder.build_graph_from_text(test_content, "test_doc_1")
        print(f"Graph build from text result: {result}")
        
        # Check statistics after building
        print("\nChecking database statistics after build...")
        result = builder.neo4j.run_query("MATCH (n) RETURN count(n) as node_count")
        record = result.single()
        new_node_count = record['node_count']
        print(f"New node count in database: {new_node_count} (added {new_node_count - node_count})")
        
        result = builder.neo4j.run_query("MATCH ()-[r]->() RETURN count(r) as rel_count")
        record = result.single()
        new_rel_count = record['rel_count']
        print(f"New relationship count in database: {new_rel_count} (added {new_rel_count - rel_count})")
    
    print("\nKnowledge Graph Test Completed Successfully!")
    
except Exception as e:
    print(f"Error during knowledge graph test: {e}")
    import traceback
    traceback.print_exc()

import asyncio
asyncio.run(main())
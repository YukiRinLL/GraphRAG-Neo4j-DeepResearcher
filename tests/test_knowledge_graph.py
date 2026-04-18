import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ms_agent.tools.knowledge_graph.knowledge_graph_tool import KnowledgeGraphTool
from ms_agent.utils import get_logger

logger = get_logger()


async def test_knowledge_graph():
    """
    Test knowledge graph functionality.
    """
    try:
        logger.info("Starting knowledge graph test...")
        
        # Initialize knowledge graph tool
        kg_tool = KnowledgeGraphTool()
        
        # Test 1: Build knowledge graph from evidence
        logger.info("Test 1: Building knowledge graph from evidence...")
        evidence_dir = "output/deep_research/benchmark_run/evidence"
        if os.path.exists(evidence_dir):
            build_result = await kg_tool.build_graph(evidence_dir)
            logger.info(f"Build result: {build_result}")
        else:
            logger.warning(f"Evidence directory not found: {evidence_dir}")
        
        # Test 2: Retrieve information from knowledge graph
        logger.info("Test 2: Retrieving information from knowledge graph...")
        retrieve_result = await kg_tool.retrieve_information("large language models", top_k=5)
        logger.info(f"Retrieve result: {retrieve_result[:200]}..." if len(retrieve_result) > 200 else f"Retrieve result: {retrieve_result}")
        
        # Test 3: Get entity connections
        logger.info("Test 3: Getting entity connections...")
        connections_result = await kg_tool.get_entity_connections("LLMs")
        logger.info(f"Connections result: {connections_result}")
        
        logger.info("Knowledge graph test completed successfully")
        return "All tests passed"
        
    except Exception as e:
        logger.error(f"Failed to test knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return f"Test failed: {str(e)}"


if __name__ == "__main__":
    result = asyncio.run(test_knowledge_graph())
    print(result)
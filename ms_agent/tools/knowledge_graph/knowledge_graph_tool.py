import os
from typing import List, Dict, Any, Optional
from ms_agent.agent.base import Agent
from ms_agent.tools.base import ToolBase
from ms_agent.llm.utils import Message, Tool
from ms_agent.utils import get_logger
from .knowledge_graph_builder import KnowledgeGraphBuilder
from .knowledge_graph_retriever import KnowledgeGraphRetriever
from .config import config

logger = get_logger()


class KnowledgeGraphTool(ToolBase):
    """
    Knowledge graph tool that provides access to knowledge graph functionality.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize knowledge graph tool.
        
        Args:
            config: Tool configuration
        """
        super().__init__(config)
        if config and hasattr(config, 'tools'):
            self.exclude_func(getattr(config.tools, 'knowledge_graph', None))
        
        try:
            self.builder = KnowledgeGraphBuilder()
            self.retriever = KnowledgeGraphRetriever()
            self.retriever.create_vector_indexes()
            logger.info("Knowledge graph tool initialized successfully")
            # Log configuration
            from .config import config as kg_config
            logger.info(f"Knowledge graph configuration: {kg_config.to_dict()}")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph tool: {e}")
            # Create dummy instances to avoid crashes
            self.builder = None
            self.retriever = None

    async def connect(self) -> None:
        """Connect the tool."""
        pass

    async def _get_tools_inner(self) -> Dict[str, Any]:
        """List tools available."""
        tools = {
            'knowledge_graph': [
                Tool(
                    tool_name='build_graph',
                    server_name='knowledge_graph',
                    description='Build knowledge graph from evidence files',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'evidence_dir': {
                                'type': 'string',
                                'description': 'Directory containing evidence files',
                            }
                        },
                        'required': ['evidence_dir'],
                        'additionalProperties': False
                    }),
                Tool(
                    tool_name='retrieve_information',
                    server_name='knowledge_graph',
                    description='Retrieve information from knowledge graph',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'Query string',
                            },
                            'top_k': {
                                'type': 'integer',
                                'description': 'Number of results to return',
                            }
                        },
                        'required': ['query'],
                        'additionalProperties': False
                    }),
                Tool(
                    tool_name='get_entity_connections',
                    server_name='knowledge_graph',
                    description='Get connections for a specific entity',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'entity_name': {
                                'type': 'string',
                                'description': 'Entity name',
                            }
                        },
                        'required': ['entity_name'],
                        'additionalProperties': False
                    }),
            ]
        }
        return tools

    async def call_tool(self, server_name: str, *, tool_name: str, tool_args: dict) -> str:
        """Call a tool."""
        return await getattr(self, tool_name)(**tool_args)

    async def build_graph(self, evidence_dir: str) -> str:
        """
        Build knowledge graph from evidence files.
        
        Args:
            evidence_dir: Directory containing evidence files
            
        Returns:
            str: Result of graph building
        """
        if not self.builder:
            return "Knowledge graph tool not initialized properly"
            
        if not os.path.exists(evidence_dir):
            return f"Evidence directory not found: {evidence_dir}"
        
        try:
            self.builder.build_from_evidence(evidence_dir)
            self.builder.update_relationships()
            return f"Knowledge graph built successfully from evidence in {evidence_dir}"
        except Exception as e:
            logger.error(f"Failed to build knowledge graph: {e}")
            return f"Failed to build knowledge graph: {str(e)}"

    async def retrieve_information(self, query: str, top_k: int = None) -> str:
        """
        Retrieve information from knowledge graph.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            str: Retrieved information
        """
        if not self.retriever:
            return "Knowledge graph tool not initialized properly"
            
        if not query:
            return "Query is required for retrieve action"
        
        if top_k is None:
            from .config import config as kg_config
            top_k = kg_config.top_k
        
        try:
            results = self.retriever.retrieve(query, top_k)
            
            # Format results
            formatted_results = []
            for result in results:
                if result['type'] == 'document':
                    formatted_results.append(
                        f"Document: {result['title']}\n"
                        f"URL: {result['url']}\n"
                        f"Content: {result['content'][:200]}...\n"
                        f"Score: {result['score']:.2f}\n"
                    )
                elif result['type'] == 'entity':
                    formatted_results.append(
                        f"Entity: {result['name']}\n"
                        f"Type: {result['type']}\n"
                        f"Context: {result['context'][:200]}...\n"
                        f"Score: {result['score']:.2f}\n"
                    )
            
            if formatted_results:
                return "\n\n".join(formatted_results)
            else:
                return "No results found in knowledge graph"
        except Exception as e:
            logger.error(f"Failed to retrieve information: {e}")
            return f"Failed to retrieve information: {str(e)}"

    async def get_entity_connections(self, entity_name: str) -> str:
        """
        Get connections for a specific entity.
        
        Args:
            entity_name: Entity name
            
        Returns:
            str: Entity connections
        """
        if not self.retriever:
            return "Knowledge graph tool not initialized properly"
            
        if not entity_name:
            return "Entity name is required for get_connections action"
        
        try:
            connections = self.retriever.get_entity_connections(entity_name)
            
            # Format connections
            formatted_connections = [f"Entity: {connections['entity']}"]
            for related in connections['related']:
                formatted_connections.append(
                    f"  - {related['name']} ({related['type']}): {related['relationship']}"
                )
            
            return "\n".join(formatted_connections)
        except Exception as e:
            logger.error(f"Failed to get entity connections: {e}")
            return f"Failed to get entity connections: {str(e)}"

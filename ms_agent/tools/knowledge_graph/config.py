# Knowledge graph configuration
import os
from typing import Dict, Any, Optional

# Load environment variables from .env file
if os.path.exists('.env'):
    with open('.env', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()


class KnowledgeGraphConfig:
    """
    Knowledge graph configuration class.
    """
    
    def __init__(self):
        """
        Initialize knowledge graph configuration.
        """
        # Neo4j configuration
        self.neo4j_uri = os.environ.get('NEO4J_URI', 'neo4j://localhost:7687')
        self.neo4j_username = os.environ.get('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
        self.neo4j_database = os.environ.get('NEO4J_DATABASE', 'neo4j')
        
        # Embedding configuration
        self.embedding_provider = os.environ.get('EMBEDDING_PROVIDER', 'local')  # local or siliconflow
        self.embedding_model = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.siliconflow_api_key = os.environ.get('SILICONFLOW_API_KEY')
        self.siliconflow_api_url = os.environ.get('SILICONFLOW_API_URL', 'https://api.siliconflow.cn/v1')
        
        # Knowledge graph configuration
        self.max_text_length = int(os.environ.get('MAX_TEXT_LENGTH', '5000'))
        self.batch_size = int(os.environ.get('BATCH_SIZE', '10'))
        self.top_k = int(os.environ.get('TOP_K', '10'))
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_username': self.neo4j_username,
            'neo4j_password': '******' if self.neo4j_password else None,
            'neo4j_database': self.neo4j_database,
            'embedding_provider': self.embedding_provider,
            'embedding_model': self.embedding_model,
            'siliconflow_api_key': '******' if self.siliconflow_api_key else None,
            'siliconflow_api_url': self.siliconflow_api_url,
            'max_text_length': self.max_text_length,
            'batch_size': self.batch_size,
            'top_k': self.top_k
        }


# Create global configuration instance
config = KnowledgeGraphConfig()

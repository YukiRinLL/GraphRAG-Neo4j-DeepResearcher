<h1> GraphRAG-Neo4j-DeepResearcher</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href='https://ms-agent-en.readthedocs.io/en/latest/'>
    <img src='https://readthedocs.org/projects/ms-agent/badge/?version=latest' alt='Documentation Status' />
</a>
<a href="https://github.com/modelscope/ms-agent/actions?query=branch%3Amaster+workflow%3Acitest++"><img src="https://img.shields.io/github/actions/workflow/status/modelscope/ms-agent/citest.yaml?branch=master&logo=github&label=CI"></a>
<a href="https://github.com/modelscope/ms-agent/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/modelscope-agent"></a>
<a href="https://github.com/modelscope/ms-agent/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href="https://pypi.org/project/ms-agent/"><img src="https://badge.fury.io/py/ms-agent.svg"></a>
<a href="https://pepy.tech/project/ms-agent"><img src="https://static.pepy.tech/badge/ms-agent"></a>
</p>



[**中文**](README_ZH.md)

## Introduction

GraphRAG-Neo4j-DeepResearcher is an enhanced deep-research agent framework that integrates MS-Agent, GraphRAG structured knowledge extraction, and Neo4j graph database for long-text understanding, multi-document reasoning, and persistent knowledge graph storage.

### Key Features Added

- **GraphRAG Integration**: Combines graph database with retrieval-augmented generation for enhanced knowledge representation and reasoning
- **Neo4j Knowledge Graph**: Persistent storage of entities, relationships, and document embeddings
- **Advanced Entity Extraction**: Multi-pattern entity recognition with quality filtering and normalization
- **Relationship Extraction**: Semantic relationship identification between entities
- **Embedding Integration**: Uses BAAI/bge-m3 model via SiliconFlow API for text embeddings
- **Configurable Knowledge Graph Tool**: Easy integration with existing MS-Agent workflows
- **Enhanced Configuration Files**: Detailed Chinese comments for all configuration options

## Project Structure

```
├── ms_agent/
│   └── tools/
│       └── knowledge_graph/     # Knowledge graph tool implementation
│           ├── __init__.py
│           ├── config.py         # Knowledge graph configuration
│           ├── knowledge_graph_builder.py  # Entity and relationship extraction
│           ├── knowledge_graph_retriever.py  # Knowledge graph retrieval
│           ├── knowledge_graph_tool.py  # Tool interface
│           └── neo4j_connection.py  # Neo4j database connection
├── projects/
│   └── deep_research/
│       └── v2/
│           ├── researcher.yaml   # Researcher agent configuration
│           ├── reporter.yaml     # Reporter agent configuration
│           └── searcher.yaml     # Searcher agent configuration
├── .env                         # Environment variables
├── run_benchmark.ps1            # Benchmark test script
└── test_kg_build.py             # Knowledge graph build test
```

## Configuration

### Environment Variables (.env)

The project uses the following environment variables for configuration:

```env
# Search API Keys
EXA_API_KEY=<EXA_API_KEY>
SERPAPI_API_KEY=<SERPAPI_API_KEY>

# LLM Configuration
OPENAI_API_KEY=<OPENAI_API_KEY>
OPENAI_BASE_URL=<OPENAI_BASE_URL>

# Neo4j Configuration
NEO4J_URI=<NEO4J_URI>
NEO4J_USERNAME=<NEO4J_USERNAME>
NEO4J_PASSWORD=<NEO4J_PASSWORD>

# Embedding Configuration
EMBEDDING_PROVIDER=<EMBEDDING_PROVIDER>
EMBEDDING_MODEL=<EMBEDDING_MODEL>
SILICONFLOW_API_KEY=<SILICONFLOW_API_KEY>
SILICONFLOW_API_URL=<SILICONFLOW_API_URL>
```

### Agent Configuration Files

The project includes three main configuration files with detailed Chinese comments:

1. **researcher.yaml**: Main orchestrator agent configuration
2. **reporter.yaml**: Report generation agent configuration  
3. **searcher.yaml**: Search and evidence collection agent configuration

Key configuration options:

- **LLM settings**: Service provider, model name, API keys
- **Tool configurations**: File system, code executor, evidence store
- **Knowledge graph integration**: Enabled via plugin system
- **Generation settings**: Stream output, prefix cache, thinking mode

## Usage

### Running the Benchmark

```powershell
# Run the deep research benchmark
.projects\deep_research\v2\run_benchmark.ps1
```

### Testing Knowledge Graph Build

```powershell
# Test knowledge graph construction
python test_kg_build.py
```

## Knowledge Graph Features

### Entity Extraction
- **Multi-pattern extraction**: Capitalized words, technical terms, acronyms, multi-word phrases
- **Entity normalization**: Consistent entity naming across documents
- **Quality filtering**: Removal of low-quality and metadata entities
- **Entity type inference**: Automatic classification of entity types

### Relationship Extraction
- **Semantic relationship identification**: Based on contextual indicators
- **Relationship quality scoring**: Filtering of irrelevant relationships
- **Relationship deduplication**: Avoiding redundant connections

### Neo4j Integration
- **Persistent storage**: Entities, relationships, and embeddings
- **Vector indexing**: For similarity-based retrieval
- **Cypher queries**: Advanced graph traversal and analysis

## Key Changes Made

1. **Added knowledge graph tool** (`ms_agent/tools/knowledge_graph/`)
2. **Enhanced entity extraction** with multiple patterns and quality filtering
3. **Integrated Neo4j** for graph storage and retrieval
4. **Added embedding support** via SiliconFlow API
5. **Updated configuration files** with Chinese comments
6. **Fixed encoding issues** with Latin-1 fallback
7. **Added benchmark test script** (`run_benchmark.ps1`)
8. **Improved error handling** for API connections

## License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

---
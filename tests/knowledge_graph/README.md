# Knowledge Graph Tests

This directory contains test scripts for the Knowledge Graph functionality.

## Test Files

### Connection Tests
- `test_neo4j_connection.py` - Tests basic Neo4j connection
- `test_neo4j_variants.py` - Tests different Neo4j URI formats
- `test_sync_neo4j.py` - Tests synchronous Neo4j connection

### Configuration Tests
- `test_config.py` - Tests configuration loading

### Component Tests
- `test_builder.py` - Tests KnowledgeGraphBuilder component
- `test_knowledge_graph.py` - Tests basic knowledge graph functionality

### Integration Tests
- `test_kg_simple.py` - Simple knowledge graph integration test
- `test_kg_complete.py` - Complete knowledge graph integration test
- `test_kg_final.py` - Final comprehensive knowledge graph test

## Running Tests

To run individual tests:
```bash
python tests/knowledge_graph/test_neo4j_connection.py
python tests/knowledge_graph/test_kg_final.py
```

## Prerequisites

1. Neo4j database must be running
2. Environment variables must be set in `.env` file:
   - `NEO4J_URI` - Neo4j connection URI
   - `NEO4J_USERNAME` - Neo4j username
   - `NEO4J_PASSWORD` - Neo4j password
   - `EMBEDDING_PROVIDER` - Embedding provider (siliconflow/local)
   - `EMBEDDING_MODEL` - Embedding model name

## Test Results

After running tests successfully, you should see:
- Successful Neo4j connection
- Knowledge graph tool initialization
- Data written to Neo4j database
- Nodes and relationships created in the graph

## Verification

To verify knowledge graph data in Neo4j Browser:
1. Open Neo4j Browser (usually http://localhost:7474)
2. Run query: `MATCH (n) RETURN n LIMIT 25`
3. Run query: `MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25`
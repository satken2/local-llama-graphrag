# Llama-based local Graph RAG example

This Python program runs Llama 3.2 3B in a local environment and creates a graph-based RAG database.

## Python programs

- graph_generation.py: Converts documents into graph data and saves it to a Neo4j graph database.
- graph_retrieval.py: Retrieves graph data related to the user's question and provides an answer.

## How to use

1. Start the Neo4j database:

```
docker compose up -d
```

2. Generate the graph database:

```
python graph_generation.py
```

3. Use the graph RAG:

```
python graph_retrieval.py
```

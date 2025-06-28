# OpenSearch Tool

A powerful Python tool for searching documents, managing indices, and retrieving data from OpenSearch clusters. Designed for use with Open WebUI and compatible with OpenSearch and Elasticsearch APIs.

## Features
- **Search Documents:** Intelligent, mapping-based query construction for natural language and advanced queries.
- **Index Management:** List, inspect, and get detailed information about indices.
- **Document Retrieval:** Fetch documents by ID with formatted output.
- **Cluster Health:** Check cluster health and connection status.
- **Automatic Field Analysis:** Dynamically analyzes index mappings to optimize queries.
- **Event Emitter Support:** Integrates with Open WebUI event emitters for real-time status updates.

## Requirements
- Python 3.7+
- [opensearch-py](https://pypi.org/project/opensearch-py/)
- Open WebUI >= 0.5.0 (for full integration)

## Installation
1. Install dependencies:
   ```bash
   pip install opensearch-py pydantic
   ```
2. Place `opensearch.py` in your Open WebUI tool directory or your project.

## Usage
### Basic Example
```python
from opensearch import Tools

tool = Tools()

# Search documents
results = await tool.search_documents(query="error logs", index="logs-*")
print(results)

# List indices
indices = await tool.list_indices()
print(indices)

# Get document by ID
result = await tool.get_document(index="logs-2024", document_id="abc123")
print(result)
```

### Configuration
You can configure connection and search parameters by editing the `Valves` class or setting attributes on `tool.valves`:
```python
tool.valves.OPENSEARCH_HOST = "http://localhost:9200"
tool.valves.OPENSEARCH_USERNAME = "admin"
tool.valves.OPENSEARCH_PASSWORD = "admin"
tool.valves.SEARCH_SIZE = 10
```

## Main Methods
- `search_documents(query, index, size=None)`: Search for documents using natural language or advanced queries.
- `list_indices()`: List all indices in the cluster.
- `get_index_info(index)`: Get detailed info about an index.
- `get_document(index, document_id)`: Retrieve a document by ID.
- `cluster_health()`: Get cluster health status.
- `test_connection()`: Test connection to the OpenSearch cluster.

## Query Examples
- `"error logs"` — Full-text search
- `"last 24 hours"` — Time-based search
- `"status:active"` — Exact match on keyword fields
- `"count > 100"` — Numeric range query
- `"recent alerts user:john"` — Combined filters

## License
MIT

## Author
Roney Dsilva

## Links
- [GitHub Repository](https://github.com/cdmx1/opensearch-mcp)
- [OpenSearch Python Client](https://opensearch.org/docs/latest/clients/python/)

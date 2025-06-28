# OpenSearch OpenWebUI Tool Fixes

## Issues Identified and Fixed

The OpenSearch OpenWebUI tool had several compatibility issues with GPT-4.1 model tool calling that were preventing it from working properly. By using the working Goodday tool as a reference, the following fixes were applied:

### 1. Function Parameter Standardization

**Problem**: OpenSearch tool functions were missing required OpenWebUI parameters.

**Fix**: Added standard OpenWebUI parameters to all async functions:
- `__request__: Request = None`
- `__user__: dict = None`  
- `__event_emitter__: Callable = None`

**Example Before**:
```python
async def search_documents(self, query: str, index: str, size: int = None, __event_emitter__=None) -> str:
```

**Example After**:
```python
async def search_documents(
    self, 
    query: str, 
    index: str, 
    size: int = None,
    __request__: Request = None,
    __user__: dict = None,
    __event_emitter__: Callable = None
) -> str:
```

### 2. Import Fixes

**Problem**: Missing required imports for proper type hints and functionality.

**Fix**: Added proper imports:
```python
import json
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from fastapi import Request
from pydantic import BaseModel, Field
from opensearchpy import OpenSearch
```

### 3. Event Emitter Formatting Consistency

**Problem**: Inconsistent event emitter formatting that could cause issues with OpenWebUI.

**Fix**: Standardized event emitter calls to match Goodday tool pattern:
```python
if __event_emitter__:
    await __event_emitter__({
        "type": "status",
        "data": {
            "description": "Operation description",
            "done": False,
        },
    })
```

### 4. Added User Agent

**Problem**: Missing user agent identification.

**Fix**: Added user agent in `__init__` method:
```python
def __init__(self):
    self.valves = self.Valves()
    self.user_agent = "opensearch-openwebui-complete/1.0.0"
```

### 5. Added Smart Query Function

**Problem**: No natural language query processing like the Goodday tool.

**Fix**: Added `opensearch_smart_query()` function that can interpret natural language requests:
- "list indices" → calls `list_indices()`
- "cluster health" → calls `get_cluster_health()`
- "search for error in logs-*" → calls `search_documents(query="error", index="logs-*")`
- And more patterns...

### 6. Error Handling Improvements

**Problem**: Inconsistent error handling and event emission.

**Fix**: Standardized error handling pattern:
```python
except Exception as e:
    error_msg = f"Error description: {str(e)}"
    if __event_emitter__:
        await __event_emitter__({
            "type": "status",
            "data": {"description": error_msg, "done": True},
        })
    return f"**Error:** {error_msg}"
```

## Functions Fixed

All the following functions were updated with proper parameters and formatting:

1. `search_documents()`
2. `list_indices()`
3. `get_index_info()`
4. `create_index()`
5. `delete_index()`
6. `get_document()`
7. `index_document()`
8. `delete_document()`
9. `delete_by_query()`
10. `get_cluster_health()`
11. `get_cluster_stats()`
12. `list_aliases()`
13. `get_alias()`
14. `put_alias()`
15. `delete_alias()`
16. `general_api_request()`

## New Function Added

- `opensearch_smart_query()` - Natural language query processing similar to Goodday tool

## Result

The OpenSearch tool should now work properly with GPT-4.1 model tool calling, matching the functionality and compatibility of the working Goodday tool.

## Testing

The tool passes Python syntax validation:
```bash
python -m py_compile opensearch.py
# Syntax OK
```

The tool should now be compatible with OpenWebUI environments and work properly with AI model tool calling.

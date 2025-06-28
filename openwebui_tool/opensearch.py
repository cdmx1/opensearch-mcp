"""
title: OpenSearch Tool
author: Roney Dsilva
git_url: https://github.com/cdmx1/opensearch-mcp
version: 1.2.1
description: OpenSearch tool for Open Web UI - search documents, manage indices, and retrieve data from OpenSearch clusters
required_open_webui_version: 0.5.0
requirements: opensearch-py
licence: MIT
"""

import json
import re
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field


def print_error(error: str) -> str:
    return f"""
```md
# Error: {error}
```"""


class Tools:
    class Valves(BaseModel):
        """Editable fields of the tool"""
        
        OPENSEARCH_HOST: str = Field(
            default="http://localhost:9200",
            description="OpenSearch endpoint URL. Default is http://localhost:9200",
        )

        OPENSEARCH_USERNAME: str = Field(
            default="admin",
            description="OpenSearch username. Default is 'admin'",
        )

        OPENSEARCH_PASSWORD: str = Field(
            default="admin",
            description="OpenSearch password. Default is 'admin'",
        )

        OPENSEARCH_PORT: int = Field(
            default=9200,
            description="OpenSearch port. Default is 9200",
        )

        USE_SSL: bool = Field(
            default=True,
            description="Whether to use SSL. Default is True",
        )

        SEARCH_SIZE: int = Field(
            default=10,
            description="Default number of search results to return",
        )

        MAX_SEARCH_SIZE: int = Field(
            default=50,
            description="Maximum number of search results to return",
        )

        DESCRIPTION: str = Field(
            default="A collection of documents stored in OpenSearch.",
            description="Description of the database. Default is 'A collection of documents stored in OpenSearch.'",
        )

    def __init__(self):
        """Initialize the tool"""
        self.valves = self.Valves()
        self._field_mappings_cache = {}

    def _get_opensearch_client(self):
        """Create and return OpenSearch client."""
        try:
            from opensearchpy import OpenSearch
        except ImportError:
            raise ImportError("opensearch-py library is required. Install with: pip install opensearch-py")
        
        # Parse host URL or use host:port format
        if self.valves.OPENSEARCH_HOST.startswith(("http://", "https://")):
            hosts = [self.valves.OPENSEARCH_HOST]
        else:
            hosts = [
                {
                    "host": self.valves.OPENSEARCH_HOST.replace("http://", "").replace("https://", ""),
                    "port": self.valves.OPENSEARCH_PORT,
                }
            ]

        return OpenSearch(
            hosts=hosts,
            http_compress=True,
            http_auth=(
                self.valves.OPENSEARCH_USERNAME,
                self.valves.OPENSEARCH_PASSWORD,
            ),
            use_ssl=self.valves.USE_SSL,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            timeout=60,
            max_retries=3,
            retry_on_timeout=True,
        )

    def _get_index_mapping(self, client, index: str) -> Dict[str, Any]:
        """Get field mappings for an index with caching."""
        if index in self._field_mappings_cache:
            return self._field_mappings_cache[index]
        
        try:
            mappings_response = client.indices.get_mapping(index=index)
            mappings = {}
            
            for idx_name, idx_data in mappings_response.items():
                if 'mappings' in idx_data and 'properties' in idx_data['mappings']:
                    mappings.update(idx_data['mappings']['properties'])
            
            self._field_mappings_cache[index] = mappings
            return mappings
            
        except Exception as e:
            # If mapping retrieval fails, return empty dict
            print(f"Warning: Could not retrieve mappings for {index}: {e}")
            return {}

    def _analyze_field_mapping(self, mapping: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze field mapping to categorize fields by type."""
        field_analysis = {
            'text_fields': [],
            'keyword_fields': [],
            'date_fields': [],
            'numeric_fields': [],
            'ip_fields': [],
            'boolean_fields': [],
            'nested_fields': [],
            'all_fields': []
        }
        
        for field_name, field_config in mapping.items():
            field_type = field_config.get('type', 'text')
            field_analysis['all_fields'].append(field_name)
            
            if field_type == 'text':
                field_analysis['text_fields'].append(field_name)
            elif field_type == 'keyword':
                field_analysis['keyword_fields'].append(field_name)
            elif field_type == 'date':
                field_analysis['date_fields'].append(field_name)
            elif field_type in ['integer', 'long', 'float', 'double', 'short', 'byte']:
                field_analysis['numeric_fields'].append(field_name)
            elif field_type == 'ip':
                field_analysis['ip_fields'].append(field_name)
            elif field_type == 'boolean':
                field_analysis['boolean_fields'].append(field_name)
            elif field_type == 'nested':
                field_analysis['nested_fields'].append(field_name)
        
        return field_analysis

    def _build_intelligent_query(self, query: str, field_analysis: Dict[str, List[str]]) -> Dict[str, Any]:
        """Build intelligent query based on field analysis and query patterns."""
        
        # Extract various patterns from query
        time_patterns = self._extract_time_patterns(query)
        numeric_patterns = self._extract_numeric_patterns(query)
        exact_patterns = self._extract_exact_patterns(query)
        
        must_clauses = []
        should_clauses = []
        filter_clauses = []
        
        # Handle time-based queries
        if time_patterns and field_analysis.get('date_fields'):
            time_query = self._build_time_query(time_patterns, field_analysis['date_fields'])
            if time_query:
                filter_clauses.append(time_query)
        
        # Handle numeric range queries
        if numeric_patterns and field_analysis.get('numeric_fields'):
            numeric_query = self._build_numeric_query(numeric_patterns, field_analysis['numeric_fields'])
            if numeric_query:
                should_clauses.extend(numeric_query)
        
        # Handle exact match patterns (field:value)
        if exact_patterns:
            exact_queries = self._build_exact_queries(exact_patterns, field_analysis)
            if exact_queries:
                must_clauses.extend(exact_queries)
        
        # Build main text search query
        text_query = self._build_text_query(query, field_analysis)
        if text_query:
            must_clauses.append(text_query)
        
        # Construct final query
        if filter_clauses or must_clauses or should_clauses:
            bool_query = {}
            
            if must_clauses:
                bool_query['must'] = must_clauses
            if should_clauses:
                bool_query['should'] = should_clauses
                bool_query['minimum_should_match'] = 1
            if filter_clauses:
                bool_query['filter'] = filter_clauses
            
            return {'bool': bool_query}
        else:
            # Fallback to match_all if no specific patterns found
            return {'match_all': {}}

    def _extract_time_patterns(self, query: str) -> List[str]:
        """Extract time-related patterns from query."""
        patterns = []
        
        # Common time expressions
        time_keywords = [
            r'last\s+(\d+)\s+(minute|hour|day|week|month)s?',
            r'past\s+(\d+)\s+(minute|hour|day|week|month)s?',
            r'recent\s+(\d+)\s+(minute|hour|day|week|month)s?',
            r'today', r'yesterday', r'this week', r'last week'
        ]
        
        for pattern in time_keywords:
            matches = re.findall(pattern, query.lower())
            patterns.extend(matches)
        
        return patterns

    def _extract_numeric_patterns(self, query: str) -> List[tuple]:
        """Extract numeric comparison patterns from query."""
        patterns = []
        
        # Patterns like "count > 100", "value < 50", "size >= 1000"
        numeric_comparisons = re.findall(r'(\w+)\s*([><=]+)\s*(\d+(?:\.\d+)?)', query)
        patterns.extend(numeric_comparisons)
        
        return patterns

    def _extract_exact_patterns(self, query: str) -> List[tuple]:
        """Extract exact match patterns like field:value."""
        patterns = []
        
        # Match patterns like "status:active", "user:john", "level:error"
        exact_matches = re.findall(r'(\w+):([^\s]+)', query)
        patterns.extend(exact_matches)
        
        return patterns

    def _build_time_query(self, time_patterns: List[str], date_fields: List[str]) -> Dict[str, Any]:
        """Build time-based query from patterns."""
        if not time_patterns or not date_fields:
            return None
        
        # Use the first date field for time queries
        date_field = date_fields[0]
        
        # For now, implement "last X hours/days" pattern
        for pattern in time_patterns:
            if isinstance(pattern, tuple) and len(pattern) == 2:
                amount, unit = pattern
                return {
                    'range': {
                        date_field: {
                            'gte': f'now-{amount}{unit[0]}/d'  # e.g., now-1d/d for last day
                        }
                    }
                }
        
        return None

    def _build_numeric_query(self, numeric_patterns: List[tuple], numeric_fields: List[str]) -> List[Dict[str, Any]]:
        """Build numeric range queries."""
        queries = []
        
        for field_name, operator, value in numeric_patterns:
            # Find matching numeric field (exact match or fuzzy match)
            target_field = None
            for num_field in numeric_fields:
                if field_name.lower() in num_field.lower() or num_field.lower() in field_name.lower():
                    target_field = num_field
                    break
            
            if target_field:
                range_query = {'range': {target_field: {}}}
                
                if '>=' in operator:
                    range_query['range'][target_field]['gte'] = float(value)
                elif '>' in operator:
                    range_query['range'][target_field]['gt'] = float(value)
                elif '<=' in operator:
                    range_query['range'][target_field]['lte'] = float(value)
                elif '<' in operator:
                    range_query['range'][target_field]['lt'] = float(value)
                elif '=' in operator:
                    range_query = {'term': {target_field: float(value)}}
                
                queries.append(range_query)
        
        return queries

    def _build_exact_queries(self, exact_patterns: List[tuple], field_analysis: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Build exact match queries for field:value patterns."""
        queries = []
        
        for field_name, value in exact_patterns:
            # Find the best matching field
            target_field = None
            
            # First try exact match in keyword fields
            if field_name in field_analysis.get('keyword_fields', []):
                target_field = field_name
                query_type = 'term'
            # Then try fuzzy match in keyword fields
            elif any(field_name.lower() in kf.lower() for kf in field_analysis.get('keyword_fields', [])):
                target_field = next(kf for kf in field_analysis.get('keyword_fields', []) if field_name.lower() in kf.lower())
                query_type = 'term'
            # Finally try text fields
            elif any(field_name.lower() in tf.lower() for tf in field_analysis.get('text_fields', [])):
                target_field = next(tf for tf in field_analysis.get('text_fields', []) if field_name.lower() in tf.lower())
                query_type = 'match_phrase'
            
            if target_field:
                if query_type == 'term':
                    queries.append({'term': {target_field: value}})
                else:
                    queries.append({'match_phrase': {target_field: value}})
        
        return queries

    def _build_text_query(self, query: str, field_analysis: Dict[str, List[str]]) -> Dict[str, Any]:
        """Build main text search query."""
        
        # Remove exact patterns from query for text search
        clean_query = re.sub(r'\w+:[^\s]+', '', query).strip()
        clean_query = re.sub(r'\w+\s*[><=]+\s*\d+(?:\.\d+)?', '', clean_query).strip()
        
        if not clean_query:
            return None
        
        # Build field list for multi_match
        search_fields = []
        
        # Prioritize text fields with boost
        if field_analysis.get('text_fields'):
            search_fields.extend([f"{field}^2" for field in field_analysis['text_fields'][:5]])
        
        # Add keyword fields
        if field_analysis.get('keyword_fields'):
            search_fields.extend(field_analysis['keyword_fields'][:3])
        
        # If no specific fields, search all
        if not search_fields:
            search_fields = ['*']
        
        return {
            'multi_match': {
                'query': clean_query,
                'fields': search_fields,
                'type': 'best_fields',
                'fuzziness': 'AUTO',
                'operator': 'or'
            }
        }

    def _explain_query_construction(self, original_query: str, field_analysis: Dict[str, List[str]], built_query: Dict[str, Any]) -> str:
        """Explain how the query was constructed."""
        
        explanation = ["**🔍 Query Analysis & Construction:**\n"]
        
        explanation.append(f"**Original Query:** `{original_query}`")
        
        # Field analysis summary
        field_summary = []
        for field_type, fields in field_analysis.items():
            if fields and field_type != 'all_fields':
                field_summary.append(f"{field_type.replace('_', ' ').title()}: {len(fields)}")
        
        explanation.append(f"**Available Fields:** {', '.join(field_summary)}")
        
        # Query structure explanation
        if 'bool' in built_query:
            bool_parts = built_query['bool']
            if 'must' in bool_parts:
                explanation.append(f"**Required Matches (MUST):** {len(bool_parts['must'])} clauses")
            if 'should' in bool_parts:
                explanation.append(f"**Optional Matches (SHOULD):** {len(bool_parts['should'])} clauses")
            if 'filter' in bool_parts:
                explanation.append(f"**Filters Applied:** {len(bool_parts['filter'])} filters")
        
        explanation.append(f"\n**Generated Query:**")
        explanation.append(f"```json\n{json.dumps(built_query, indent=2)}\n```")
        
        return "\n".join(explanation)

    def _build_dynamic_query(self, query: str, mappings: Dict[str, Any]) -> Dict[str, Any]:
        """Build dynamic DSL query based on field mappings and query content."""
        
        # Analyze query for different patterns
        date_patterns = re.findall(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b', query)
        number_patterns = re.findall(r'\b\d+\.?\d*\b', query)
        ip_patterns = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', query)
        email_patterns = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', query)
        
        # Categorize fields by type
        text_fields = []
        keyword_fields = []
        date_fields = []
        numeric_fields = []
        ip_fields = []
        
        for field_name, field_config in mappings.items():
            field_type = field_config.get('type', 'text')
            
            if field_type in ['text']:
                text_fields.append(field_name)
            elif field_type in ['keyword']:
                keyword_fields.append(field_name)
            elif field_type in ['date']:
                date_fields.append(field_name)
            elif field_type in ['integer', 'long', 'float', 'double']:
                numeric_fields.append(field_name)
            elif field_type in ['ip']:
                ip_fields.append(field_name)
        
        # Build dynamic query based on detected patterns and field types
        must_clauses = []
        should_clauses = []
        
        # Handle date patterns
        if date_patterns and date_fields:
            for date_val in date_patterns[:2]:  # Limit to 2 dates
                date_range = {
                    "range": {
                        date_fields[0]: {
                            "gte": date_val,
                            "lte": date_val,
                            "format": "yyyy-MM-dd||dd/MM/yyyy"
                        }
                    }
                }
                should_clauses.append(date_range)
        
        # Handle IP patterns
        if ip_patterns and ip_fields:
            for ip_val in ip_patterns:
                ip_query = {
                    "term": {
                        ip_fields[0]: ip_val
                    }
                }
                should_clauses.append(ip_query)
        
        # Handle email patterns
        if email_patterns:
            for email in email_patterns:
                email_query = {
                    "multi_match": {
                        "query": email,
                        "fields": keyword_fields + text_fields if keyword_fields or text_fields else ["*"],
                        "type": "phrase"
                    }
                }
                should_clauses.append(email_query)
        
        # Handle numeric patterns
        if number_patterns and numeric_fields:
            for num_val in number_patterns[:3]:  # Limit to 3 numbers
                try:
                    num_float = float(num_val)
                    numeric_query = {
                        "range": {
                            numeric_fields[0]: {
                                "gte": num_float * 0.9,  # Allow 10% variance
                                "lte": num_float * 1.1
                            }
                        }
                    }
                    should_clauses.append(numeric_query)
                except ValueError:
                    continue
        
        # Main text search across appropriate fields
        search_fields = []
        if text_fields:
            search_fields.extend([f"{field}^2" for field in text_fields[:5]])  # Boost text fields
        if keyword_fields:
            search_fields.extend(keyword_fields[:3])
        
        if not search_fields:
            search_fields = ["*"]
        
        # Clean query for text search (remove patterns already handled)
        clean_query = query
        for pattern in date_patterns + ip_patterns + email_patterns:
            clean_query = clean_query.replace(pattern, "").strip()
        
        if clean_query:
            main_search = {
                "multi_match": {
                    "query": clean_query,
                    "fields": search_fields,
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "operator": "or"
                }
            }
            must_clauses.append(main_search)
        
        # Build final query structure
        if should_clauses and must_clauses:
            final_query = {
                "bool": {
                    "must": must_clauses,
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
        elif must_clauses:
            final_query = {
                "bool": {
                    "must": must_clauses
                }
            }
        elif should_clauses:
            final_query = {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
        else:
            # Fallback to simple multi_match
            final_query = {
                "multi_match": {
                    "query": query,
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
        
        return final_query

    def _get_index_mapping(self, client, index: str) -> Dict[str, Any]:
        """Get the mapping for the specified index (synchronous version)."""
        try:
            mapping = client.indices.get_mapping(index=index)
            return mapping
        except Exception as e:
            print(f"Warning: Could not get mapping for index {index}: {e}")
            return {}

    def _analyze_field_mapping(self, mapping: dict) -> dict:
        """
        Analyze the index mapping to understand field types and their search characteristics.
        
        Returns a dictionary with field analysis for optimized query construction.
        """
        field_analysis = {
            'text_fields': [],      # Full-text searchable fields
            'keyword_fields': [],   # Exact match fields
            'numeric_fields': [],   # Numeric fields (int, long, float, double)
            'date_fields': [],      # Date/timestamp fields
            'boolean_fields': [],   # Boolean fields
            'nested_fields': [],    # Nested object fields
            'analyzed_fields': {},  # Field -> analyzer mapping
        }
        
        def analyze_properties(properties: dict, parent_path: str = ""):
            for field_name, field_config in properties.items():
                full_field_name = f"{parent_path}.{field_name}" if parent_path else field_name
                field_type = field_config.get('type', '')
                
                if field_type == 'text':
                    field_analysis['text_fields'].append(full_field_name)
                    analyzer = field_config.get('analyzer', 'standard')
                    field_analysis['analyzed_fields'][full_field_name] = analyzer
                elif field_type == 'keyword':
                    field_analysis['keyword_fields'].append(full_field_name)
                elif field_type in ['integer', 'long', 'float', 'double', 'short', 'byte']:
                    field_analysis['numeric_fields'].append(full_field_name)
                elif field_type == 'date':
                    field_analysis['date_fields'].append(full_field_name)
                elif field_type == 'boolean':
                    field_analysis['boolean_fields'].append(full_field_name)
                elif field_type == 'nested':
                    field_analysis['nested_fields'].append(full_field_name)
                    # Recursively analyze nested properties
                    nested_props = field_config.get('properties', {})
                    if nested_props:
                        analyze_properties(nested_props, full_field_name)
                elif field_type == 'object':
                    # Recursively analyze object properties
                    object_props = field_config.get('properties', {})
                    if object_props:
                        analyze_properties(object_props, full_field_name)
                
                # Check for multi-fields (fields with .keyword, .raw, etc.)
                fields = field_config.get('fields', {})
                for sub_field_name, sub_field_config in fields.items():
                    sub_full_name = f"{full_field_name}.{sub_field_name}"
                    sub_type = sub_field_config.get('type', '')
                    if sub_type == 'keyword':
                        field_analysis['keyword_fields'].append(sub_full_name)
        
        # Extract properties from mapping
        for index_name, index_mapping in mapping.items():
            mappings = index_mapping.get('mappings', {})
            properties = mappings.get('properties', {})
            analyze_properties(properties)
        
        return field_analysis

    def _build_intelligent_query(self, query_text: str, field_analysis: dict) -> dict:
        """
        Build an intelligent OpenSearch DSL query based on the query text and field analysis.
        
        This function analyzes the query and constructs the most appropriate query type
        based on the available fields and their types.
        """
        query_lower = query_text.lower().strip()
        
        # Check for time-based queries
        time_keywords = ["last", "latest", "recent", "new", "current", "today", "yesterday"]
        is_time_query = any(keyword in query_lower for keyword in time_keywords)
        
        # Check for numeric queries (ranges, comparisons)
        numeric_patterns = [
            r'\b(greater|less|more|fewer)\s+than\s+(\d+)',
            r'\b(>=|<=|>|<|=)\s*(\d+)',
            r'\bbetween\s+(\d+)\s+and\s+(\d+)',
            r'\b(\d+)\s*-\s*(\d+)\b'
        ]
        numeric_matches = []
        for pattern in numeric_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                numeric_matches.extend(matches)
        
        # Check for exact match indicators
        exact_match_indicators = ['"', "'", "exact", "exactly", "id:", "user:", "status:"]
        is_exact_query = any(indicator in query_lower for indicator in exact_match_indicators)
        
        # Extract quoted phrases for exact matching
        quoted_phrases = re.findall(r'"([^"]+)"', query_text)
        quoted_phrases.extend(re.findall(r"'([^']+)'", query_text))
        
        # Build the query
        query_clauses = []
        
        # Handle time-based queries
        if is_time_query and field_analysis.get('date_fields'):
            # Add time range query
            date_field = field_analysis['date_fields'][0]  # Use first available date field
            
            if "today" in query_lower:
                query_clauses.append({
                    "range": {
                        date_field: {
                            "gte": "now/d",
                            "lt": "now+1d/d"
                        }
                    }
                })
            elif "yesterday" in query_lower:
                query_clauses.append({
                    "range": {
                        date_field: {
                            "gte": "now-1d/d",
                            "lt": "now/d"
                        }
                    }
                })
            elif any(kw in query_lower for kw in ["last", "recent"]):
                # Extract time period (e.g., "last 24 hours", "recent 7 days")
                time_match = re.search(r'(last|recent)\s+(\d+)\s*(hour|day|week|month)', query_lower)
                if time_match:
                    number = int(time_match.group(2))
                    unit = time_match.group(3)
                    unit_map = {"hour": "h", "day": "d", "week": "w", "month": "M"}
                    query_clauses.append({
                        "range": {
                            date_field: {
                                "gte": f"now-{number}{unit_map.get(unit, 'd')}"
                            }
                        }
                    })
                else:
                    # Default to last 24 hours
                    query_clauses.append({
                        "range": {
                            date_field: {
                                "gte": "now-24h"
                            }
                        }
                    })
        
        # Handle numeric queries
        if numeric_matches and field_analysis.get('numeric_fields'):
            for match in numeric_matches:
                if len(match) >= 2:
                    operator, value = match[0], match[1]
                    numeric_field = field_analysis['numeric_fields'][0]  # Use first numeric field
                    
                    if operator in ['greater', 'more', '>', '>=']:
                        query_clauses.append({
                            "range": {
                                numeric_field: {
                                    "gte" if '>=' in operator else "gt": int(value)
                                }
                            }
                        })
                    elif operator in ['less', 'fewer', '<', '<=']:
                        query_clauses.append({
                            "range": {
                                numeric_field: {
                                    "lte" if '<=' in operator else "lt": int(value)
                                }
                            }
                        })
        
        # Handle exact match queries
        if quoted_phrases and field_analysis.get('keyword_fields'):
            for phrase in quoted_phrases:
                # Try to match against keyword fields first
                keyword_query = {
                    "multi_match": {
                        "query": phrase,
                        "fields": field_analysis['keyword_fields'][:5],  # Limit to first 5 keyword fields
                        "type": "phrase"
                    }
                }
                query_clauses.append(keyword_query)
        
        # Handle general text search
        remaining_text = query_text
        for phrase in quoted_phrases:
            remaining_text = remaining_text.replace(f'"{phrase}"', '').replace(f"'{phrase}'", '')
        
        remaining_text = ' '.join(remaining_text.split())  # Clean up whitespace
        
        if remaining_text and field_analysis.get('text_fields'):
            # Use multi_match for text fields
            text_query = {
                "multi_match": {
                    "query": remaining_text,
                    "fields": field_analysis['text_fields'][:10],  # Limit to first 10 text fields
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "operator": "and" if len(remaining_text.split()) <= 2 else "or"
                }
            }
            query_clauses.append(text_query)
        elif remaining_text:
            # Fallback: search all fields
            fallback_query = {
                "query_string": {
                    "query": remaining_text,
                    "default_operator": "AND",
                    "fuzziness": "AUTO"
                }
            }
            query_clauses.append(fallback_query)
        
        # Combine queries
        if len(query_clauses) == 0:
            return {"match_all": {}}
        elif len(query_clauses) == 1:
            return query_clauses[0]
        else:
            return {
                "bool": {
                    "must": query_clauses
                }
            }

    def _explain_query_construction(self, original_query: str, field_analysis: dict, constructed_query: dict) -> str:
        """
        Provide an explanation of how the query was constructed based on the analysis.
        """
        explanations = []
        
        # Analyze what was detected in the query
        query_lower = original_query.lower()
        
        if any(kw in query_lower for kw in ["last", "latest", "recent", "today", "yesterday"]):
            explanations.append("🕒 **Time-based query detected** - Applied date range filters")
        
        if any(op in query_lower for op in ["greater", "less", "more", "fewer", ">", "<", "="]):
            explanations.append("🔢 **Numeric comparison detected** - Applied range filters")
        
        if '"' in original_query or "'" in original_query:
            explanations.append("🎯 **Exact phrase matching** - Using keyword fields for precise matches")
        
        # Provide field analysis summary
        field_summary = []
        if field_analysis.get('text_fields'):
            field_summary.append(f"{len(field_analysis['text_fields'])} text fields")
        if field_analysis.get('keyword_fields'):
            field_summary.append(f"{len(field_analysis['keyword_fields'])} keyword fields")
        if field_analysis.get('date_fields'):
            field_summary.append(f"{len(field_analysis['date_fields'])} date fields")
        if field_analysis.get('numeric_fields'):
            field_summary.append(f"{len(field_analysis['numeric_fields'])} numeric fields")
        
        if field_summary:
            explanations.append(f"📊 **Index analysis** - Found {', '.join(field_summary)}")
        
        # Show query type
        if 'bool' in constructed_query:
            explanations.append("🔍 **Complex query** - Combined multiple search criteria")
        elif 'multi_match' in constructed_query:
            explanations.append("📝 **Text search** - Searching across relevant text fields")
        elif 'range' in constructed_query:
            explanations.append("📅 **Range query** - Filtering by date/numeric ranges")
        elif 'match_all' in constructed_query:
            explanations.append("📋 **Retrieve all** - Getting latest documents")
        
        if explanations:
            return "**Query Analysis:**\n" + "\n".join(f"- {exp}" for exp in explanations)
        else:
            return "**Query Analysis:** Standard text search applied"

    def _explain_query_strategy(self, query: str, mappings: Dict[str, Any], built_query: Dict[str, Any]) -> str:
        """Explain the query strategy used."""
        explanation = ["**Query Strategy Explanation:**\n"]
        
        # Analyze what was detected
        date_patterns = re.findall(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b', query)
        number_patterns = re.findall(r'\b\d+\.?\d*\b', query)
        ip_patterns = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', query)
        email_patterns = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', query)
        
        explanation.append(f"- **Original Query:** `{query}`")
        explanation.append(f"- **Detected Patterns:** Dates: {len(date_patterns)}, IPs: {len(ip_patterns)}, Emails: {len(email_patterns)}, Numbers: {len(number_patterns)}")
        
        # Field analysis
        field_types = {}
        for field_name, field_config in mappings.items():
            field_type = field_config.get('type', 'text')
            if field_type not in field_types:
                field_types[field_type] = []
            field_types[field_type].append(field_name)
        
        explanation.append(f"- **Available Field Types:** {', '.join([f'{k}({len(v)})' for k, v in field_types.items()])}")
        
        # Query structure explanation
        if "bool" in built_query:
            bool_query = built_query["bool"]
            if "must" in bool_query:
                explanation.append(f"- **MUST clauses:** {len(bool_query['must'])} (required matches)")
            if "should" in bool_query:
                explanation.append(f"- **SHOULD clauses:** {len(bool_query['should'])} (optional matches)")
        
        explanation.append(f"- **Generated DSL Query:**")
        explanation.append(f"```json\n{json.dumps(built_query, indent=2)}\n```")
        
        return "\n".join(explanation)

    def _format_search_results(self, results: Dict[str, Any]) -> str:
        """Format search results for display"""
        if not results or "hits" not in results:
            return "No results found."

        hits = results["hits"]["hits"]
        total = results["hits"]["total"]

        if isinstance(total, dict):
            total_count = total.get("value", 0)
        else:
            total_count = total

        formatted_results = [f"**Found {total_count:,} total results, showing {len(hits)} documents:**\n"]

        for i, hit in enumerate(hits, 1):
            source = hit.get("_source", {})
            score = hit.get("_score", "N/A")
            doc_id = hit.get("_id", "N/A")
            index = hit.get("_index", "N/A")

            formatted_results.append(f"### Result {i}")
            formatted_results.append(f"**Index:** {index}")
            formatted_results.append(f"**Document ID:** {doc_id}")
            formatted_results.append(f"**Score:** {score}")
            formatted_results.append("**Source:**")
            formatted_results.append(f"```json\n{json.dumps(source, indent=2)}\n```")
            formatted_results.append("")

        return "\n".join(formatted_results)



    async def get_index_info(self, index: str) -> str:
        """
        Get detailed information about a specific index.
        
        Args:
            index: Name of the index to get information about
        """
        try:
            client = self._get_opensearch_client()
            
            # Get index stats
            stats = client.indices.stats(index=index)
            
            # Get index mappings
            mappings = client.indices.get_mapping(index=index)
            
            # Get index settings
            settings = client.indices.get_settings(index=index)
            
            result = [f"**Index Information: {index}**\n"]
            
            # Basic stats
            if stats and 'indices' in stats:
                index_stats = stats['indices'].get(index, {})
                total_docs = index_stats.get('total', {}).get('docs', {}).get('count', 0)
                total_size = index_stats.get('total', {}).get('store', {}).get('size_in_bytes', 0)
                
                result.append(f"**Document Count:** {total_docs:,}")
                result.append(f"**Size:** {total_size:,} bytes ({total_size / (1024*1024):.2f} MB)")
            
            # Field mappings
            result.append("\n**Field Mappings:**")
            for idx_name, idx_data in mappings.items():
                if 'mappings' in idx_data and 'properties' in idx_data['mappings']:
                    properties = idx_data['mappings']['properties']
                    for field_name, field_config in properties.items():
                        field_type = field_config.get('type', 'unknown')
                        result.append(f"- `{field_name}`: {field_type}")
            
            return "\n".join(result)
            
        except Exception as e:
            return print_error(f"Failed to get index info: {str(e)}")

    async def get_document_by_id(self, index: str, doc_id: str) -> str:
        """
        Retrieve a specific document by its ID.
        
        Args:
            index: Index name
            doc_id: Document ID
        """
        try:
            client = self._get_opensearch_client()
            
            result = client.get(index=index, id=doc_id)
            
            if result and '_source' in result:
                formatted_result = [
                    f"**Document found in index: {index}**\n",
                    f"**Document ID:** {result.get('_id', 'N/A')}",
                    f"**Version:** {result.get('_version', 'N/A')}",
                    f"**Source:**",
                    f"```json\n{json.dumps(result['_source'], indent=2)}\n```"
                ]
                return "\n".join(formatted_result)
            else:
                return f"Document with ID '{doc_id}' not found in index '{index}'"
                
        except Exception as e:
            return print_error(f"Failed to get document: {str(e)}")

    async def cluster_health(self) -> str:
        """
        Get OpenSearch cluster health information.
        """
        try:
            client = self._get_opensearch_client()
            
            health = client.cluster.health()
            
            result = [
                "**OpenSearch Cluster Health:**\n",
                f"**Status:** {health.get('status', 'unknown')}",
                f"**Cluster Name:** {health.get('cluster_name', 'unknown')}",
                f"**Number of Nodes:** {health.get('number_of_nodes', 0)}",
                f"**Number of Data Nodes:** {health.get('number_of_data_nodes', 0)}",
                f"**Active Primary Shards:** {health.get('active_primary_shards', 0)}",
                f"**Active Shards:** {health.get('active_shards', 0)}",
                f"**Relocating Shards:** {health.get('relocating_shards', 0)}",
                f"**Initializing Shards:** {health.get('initializing_shards', 0)}",
                f"**Unassigned Shards:** {health.get('unassigned_shards', 0)}",
            ]
            
            return "\n".join(result)
            
        except Exception as e:
            return print_error(f"Failed to get cluster health: {str(e)}")

    async def search_documents(self, query: str, index: str, size: int = None, __event_emitter__=None) -> str:
        """
        Search for documents in OpenSearch indices using intelligent mapping-based query construction.

        This function automatically:
        1. Fetches the index mapping to understand field types
        2. Analyzes the query text for patterns (time ranges, numeric comparisons, exact matches)
        3. Constructs an optimized OpenSearch DSL query based on field types and query intent
        4. Applies appropriate sorting for time-based queries

        User Intent Examples:
        - "error logs" → searches text fields for "error logs"
        - "last 24 hours" → adds time range filter on date fields
        - "status: active" → exact match on keyword fields
        - "count > 100" → numeric range query on numeric fields
        - "recent alerts user:john" → combines time filter + exact user match + text search
        
        :param query: Search query string with natural language patterns
        :param index: Index name/pattern to search (REQUIRED - e.g., "logs-2024", "alerts-*", "events-*")
        :param size: Number of results to return (max 50 for performance)
        :param __event_emitter__: Optional event emitter for handling events
        :return: The search results as a string
        """
        try:
            if not index:
                return "**Error:** Index parameter is required. Please specify which index to search (e.g., 'logs-*', 'events-2024', 'alerts-*')"
            
            if not query:
                return "**Error:** Query parameter is required. Please specify what to search for."
            
            if not size:
                size = self.valves.SEARCH_SIZE

            # Enforce maximum size limit for performance
            size = min(size, self.valves.MAX_SEARCH_SIZE)

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Connecting to OpenSearch cluster...",
                        "done": False,
                    },
                })

            client = self._get_opensearch_client()
            
            # Test connection first
            try:
                cluster_health = client.cluster.health()
            except Exception as e:
                return f"**Error:** Could not connect to OpenSearch cluster: {str(e)}"
            
            # Get index mapping to understand field types
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Analyzing index mapping for '{index}'...",
                        "done": False,
                    },
                })

            mapping = self._get_index_mapping(client, index)
            field_analysis = self._analyze_field_mapping(mapping)

            # Build intelligent query based on mapping and query text
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Building optimized search query...",
                        "done": False,
                    },
                })

            intelligent_query = self._build_intelligent_query(query, field_analysis)

            # Detect if user wants recent/latest records for sorting
            time_keywords = ["last", "latest", "recent", "new", "current", "today", "yesterday"]
            wants_recent = any(keyword in query.lower() for keyword in time_keywords)

            # Extract number if present (e.g., "last 10" → size=10)
            if wants_recent:
                number_match = re.search(r"\b(?:last|latest|recent)\s+(\d+)\b", query.lower())
                if number_match:
                    requested_size = min(int(number_match.group(1)), self.valves.MAX_SEARCH_SIZE)
                    size = requested_size

            # Create search body with intelligent query
            search_body = {
                "size": size,
                "query": intelligent_query,
                "timeout": "60s",
            }

            # Add intelligent sorting based on available fields
            sort_fields = []
            
            if wants_recent and field_analysis.get('date_fields'):
                # Sort by available date fields in descending order
                for date_field in field_analysis['date_fields'][:3]:  # Use up to 3 date fields
                    sort_fields.append({date_field: {"order": "desc", "missing": "_last"}})
            
            # Always add score as final sort criterion
            sort_fields.append({"_score": {"order": "desc"}})
            search_body["sort"] = sort_fields

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Executing optimized search on index '{index}' (max {size} results)",
                        "done": False,
                    },
                })

            # Execute the search
            results = client.search(
                index=index,
                body=search_body,
                request_timeout=60,
            )

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Search completed successfully",
                        "done": True,
                    },
                })

            # Add query explanation to the results
            query_explanation = self._explain_query_construction(query, field_analysis, intelligent_query)
            formatted_results = self._format_search_results(results)
            
            return f"{query_explanation}\n\n{formatted_results}"

        except Exception as e:
            error_msg = f"Error searching documents: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "message",
                    "data": {
                        "content": print_error(error_msg)
                    },
                })
            return f"**Error:** {error_msg}"

    async def list_indices(self, __event_emitter__=None) -> str:
        """
        List all available indices in the OpenSearch cluster.
        
        :param __event_emitter__: Optional event emitter for handling events
        :return: List of indices as a string
        """
        try:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Retrieving list of indices...",
                        "done": False,
                    },
                })

            client = self._get_opensearch_client()
            indices = client.cat.indices(format="json")

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Found {len(indices)} indices",
                        "done": True,
                    },
                })

            formatted_indices = [f"**Available Indices ({len(indices)} total):**\n"]

            for idx in indices:
                name = idx.get("index", "Unknown")
                health = idx.get("health", "Unknown")
                docs_count = idx.get("docs.count", "Unknown")
                size_str = idx.get("store.size", "Unknown")
                status = idx.get("status", "Unknown")

                # Add status indicator
                status_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(health, "⚪")

                formatted_indices.append(
                    f"- {status_emoji} **{name}** (Health: {health}, Status: {status}, Docs: {docs_count}, Size: {size_str})"
                )

            return "\n".join(formatted_indices)

        except Exception as e:
            error_msg = f"Error listing indices: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "message",
                    "data": {
                        "content": print_error(error_msg)
                    },
                })
            return f"**Error:** {error_msg}"

    async def get_index_info(self, index: str, __event_emitter__=None) -> str:
        """
        Get detailed information about a specific index.
        
        :param index: Name of the index to get information about
        :param __event_emitter__: Optional event emitter for handling events
        :return: Index information as a string
        """
        try:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Retrieving information for index '{index}'...",
                        "done": False,
                    },
                })

            client = self._get_opensearch_client()
            info = client.indices.get(index=index)

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Retrieved index information successfully",
                        "done": True,
                    },
                })

            return f"**Index Information for '{index}':**\n```json\n{json.dumps(info, indent=2)}\n```"

        except Exception as e:
            error_msg = f"Error getting index info: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "message",
                    "data": {
                        "content": print_error(error_msg)
                    },
                })
            return f"**Error:** {error_msg}"

    async def get_document(self, index: str, document_id: str, __event_emitter__=None) -> str:
        """
        Get a specific document by ID.
        
        :param index: Name of the index
        :param document_id: ID of the document to retrieve
        :param __event_emitter__: Optional event emitter for handling events
        :return: Document data as a string
        """
        try:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Retrieving document '{document_id}' from index '{index}'...",
                        "done": False,
                    },
                })

            client = self._get_opensearch_client()
            result = client.get(index=index, id=document_id)

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Document retrieved successfully",
                        "done": True,
                    },
                })

            return f"**Document '{document_id}' from index '{index}':**\n```json\n{json.dumps(result, indent=2)}\n```"

        except Exception as e:
            error_msg = f"Error getting document: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "message",
                    "data": {
                        "content": print_error(error_msg)
                    },
                })
            return f"**Error:** {error_msg}"

    async def test_connection(self, __event_emitter__=None) -> str:
        """
        Test the connection to OpenSearch cluster.
        
        :param __event_emitter__: Optional event emitter for handling events
        :return: Connection test results as a string
        """
        try:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "Testing OpenSearch connection...",
                        "done": False,
                    },
                })

            client = self._get_opensearch_client()
            health = client.cluster.health()
            info = client.info()

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Connection test completed successfully",
                        "done": True,
                    },
                })

            return f"""**OpenSearch Connection Test - SUCCESS ✅**

**Cluster Health:**
- Status: {health.get('status', 'Unknown')}
- Cluster Name: {health.get('cluster_name', 'Unknown')}
- Number of Nodes: {health.get('number_of_nodes', 'Unknown')}
- Active Shards: {health.get('active_shards', 'Unknown')}

**Server Info:**
- Version: {info.get('version', {}).get('number', 'Unknown')}
- Build: {info.get('version', {}).get('build_type', 'Unknown')}

Connection to OpenSearch cluster is working properly."""

        except Exception as e:
            error_msg = f"Connection test failed: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "message",
                    "data": {
                        "content": print_error(error_msg)
                    },
                })
            return f"**OpenSearch Connection Test - FAILED ❌**\n\n**Error:** {error_msg}"

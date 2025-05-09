"""
tRNA Ontology Retriever - Async Version

This system combines vector embeddings for semantic field retrieval with
path traversal to extract relevant subgraphs from the tRNA database ontology.
"""

import json
import numpy as np
import logging
import asyncio
import time
import os        
import aiosqlite
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
import voyageai
import httpx
from dotenv import load_dotenv

# LiteLLM configuration
load_dotenv()
LITELLM_API_KEY = os.getenv('LITELLM_API_KEY')
LITELLM_BASE_URL = os.getenv('LITELLM_BASE_URL')

# Database configuration
DB_PATH = os.getenv('TRNA_DB_PATH', 'trna_db_v01.db')

# Ollama configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

# Available models
AVAILABLE_MODELS = [
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
    "openai-gpt-3.5",
    "openai-gpt-4",
    "R1",
    "mistral-24b",
    "qwen-72b-instruct",
    "o3-mini-high",
    "flash-2",
    "sonar",
    "Ministral-8b",
    # Ollama model
    "ollama:mistral-openorca"
]

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector (or first batch of vectors)
        b: Second vector (or second batch of vectors)
        
    Returns:
        Cosine similarity between vectors
    """
    # Handle batch calculation if a is a batch (2D array)
    if len(a.shape) > 1 and a.shape[0] > 1:
        # Normalize vectors
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        
        # Avoid division by zero
        a_norm = np.maximum(a_norm, np.finfo(a.dtype).eps)
        b_norm = np.maximum(b_norm, np.finfo(b.dtype).eps)
        
        # Calculate cosine similarity
        return np.dot(a / a_norm, (b / b_norm).T)
    else:
        # Single vector calculation
        # Normalize vectors
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        # Avoid division by zero
        a_norm = max(a_norm, np.finfo(a.dtype).eps)
        b_norm = max(b_norm, np.finfo(b.dtype).eps)
        
        # Calculate cosine similarity
        return np.dot(a / a_norm, (b / b_norm).T)

class AsyncTRNARetriever:
    def __init__(self, embedding_file: str, ontology_file: str, model: str = "voyage-3-large"):
        """
        Initialize the retriever with the embedding index and ontology files.
        
        Args:
            embedding_file: Path to the field embeddings JSON file
            ontology_file: Path to the full ontology JSON file
            model: Embedding model to use for semantic search
        """
        self.embedding_model = model
        self.vo = voyageai.Client()
        
        # Load the embedding index
        with open(embedding_file, 'r') as f:
            self.embedding_index = json.load(f)
        
        # Load the full ontology
        with open(ontology_file, 'r') as f:
            self.ontology = json.load(f)
        
        # Define field associations - when a field is targeted, include these additional fields
        self.field_associations = {
            "modification.trna_id": ["modification.name", "modification.position"],
            "trna.id": ["trna.anticodon", "trna.isotype"],
            # Additional associations can be defined here
        }
        
        # Extract embedding vectors for faster retrieval
        self.field_ids = []
        self.embedding_vectors = []
        
        for field_id, field_data in self.embedding_index["embeddings"].items():
            self.field_ids.append(field_id)
            self.embedding_vectors.append(field_data["embedding"])
        
        self.embedding_vectors = np.array(self.embedding_vectors)
        
        logging.info(f"Loaded {len(self.field_ids)} field embeddings")
        logging.info(f"Loaded ontology with {len(self.ontology['entities'])} entities and {len(self.ontology['relationships'])} relationships")

    async def embed_query(self, query_text: str) -> np.ndarray:
        """
        Embed a query string using the Voyage API.
        
        Args:
            query_text: The query text to embed
            
        Returns:
            The query embedding vector
        """
        # Create an async wrapper around the synchronous API call
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: self.vo.embed(query_text, model=self.embedding_model)
        )
        embedding = np.array(response.embeddings[0])
        return embedding

    async def retrieve_top_k_fields(self, query_text: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve the top-k most semantically similar fields to the query.
        
        Args:
            query_text: The query text
            k: Number of fields to retrieve
            
        Returns:
            List of (field_id, similarity_score) tuples
        """
        # Embed the query
        query_vector = await self.embed_query(query_text)
        
        # Calculate cosine similarity with all field embeddings
        similarities = cosine_similarity(query_vector.reshape(1, -1), self.embedding_vectors)[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-k:][::-1]
        
        # Return field IDs and similarity scores
        results = [(self.field_ids[idx], similarities[idx]) for idx in top_indices]
        return results

    def extract_field_info(self, field_id: str) -> Dict[str, Any]:
        """
        Extract detailed information about a field from the ontology.
        
        Args:
            field_id: The field ID in format "table.field"
            
        Returns:
            Dictionary with field information
        """
        table, field = field_id.split('.')
        
        # Find the entity that contains this table
        entity_name = None
        for name, entity in self.ontology["entities"].items():
            if entity.get("table") == table:
                entity_name = name
                break
        
        if not entity_name:
            return {"error": f"Table {table} not found in ontology"}
        
        # Get field information
        if field not in self.ontology["entities"][entity_name]["fields"]:
            return {"error": f"Field {field} not found in {entity_name}"}
        
        field_info = self.ontology["entities"][entity_name]["fields"][field].copy()
        
        # Add entity context
        field_info["entity"] = entity_name
        field_info["table"] = table
        field_info["field"] = field
        field_info["field_id"] = field_id
        
        return field_info

    def extract_minimal_subgraph(self, target_fields: List[str]) -> Dict[str, Any]:
        """
        Extract a truly minimal subgraph that connects the target fields.
        Only includes:
        1. The specific target fields themselves and their associated fields
        2. The fields needed to establish relationships between entities
        3. The minimal path connecting these fields (no duplicates)
        4. Always includes species table with scientific_name and genome_assembly fields
        """
        result = {
            "nodes": {},
            "edges": []
        }
        
        # Expand target fields to include associated fields
        expanded_target_fields = set(target_fields)
        for field_id in target_fields:
            if field_id in self.field_associations:
                expanded_target_fields.update(self.field_associations[field_id])
        
        # First, identify target tables and add target fields
        target_entities = set()
        
        for field_id in expanded_target_fields:
            table, field = field_id.split('.')
            
            # Find the entity for this table
            entity_name = None
            for name, entity in self.ontology["entities"].items():
                if entity.get("table") == table:
                    entity_name = name
                    break
            
            if not entity_name or field not in self.ontology["entities"][entity_name]["fields"]:
                continue
            
            # Add to target entities list
            target_entities.add(entity_name)
            
            # Special case for junction tables
            if entity_name == "tRNA_modification_association":
                target_entities.add("Modification") 
                target_entities.add("trna")
            
            # Add entity and the target field
            if entity_name not in result["nodes"]:
                result["nodes"][entity_name] = {
                    "table": self.ontology["entities"][entity_name]["table"],
                    "description": self.ontology["entities"][entity_name].get("description", ""),
                    "fields": {}
                }
            
            # Add the target field
            result["nodes"][entity_name]["fields"][field] = self.ontology["entities"][entity_name]["fields"][field]
        
        # Always add the "trna" entity if not already included
        if "trna" not in result["nodes"] and "trna" in self.ontology["entities"]:
            result["nodes"]["trna"] = {
                "table": self.ontology["entities"]["trna"]["table"],
                "description": self.ontology["entities"]["trna"].get("description", ""),
                "fields": {}
            }
            
            # Add species_id field to the trna entity
            if "species_id" in self.ontology["entities"]["trna"]["fields"]:
                result["nodes"]["trna"]["fields"]["species_id"] = self.ontology["entities"]["trna"]["fields"]["species_id"]
                
            # Make sure trna is in target entities
            target_entities.add("trna")
        
        # Always add the "species" entity and required fields
        if "species" in self.ontology["entities"]:
            result["nodes"]["species"] = {
                "table": self.ontology["entities"]["species"]["table"],
                "description": self.ontology["entities"]["species"].get("description", ""),
                "fields": {}
            }
            
            # Ensure species has required fields
            required_fields = ["scientific_name", "genome_assembly", "id"]
            for field in required_fields:
                if field in self.ontology["entities"]["species"]["fields"]:
                    result["nodes"]["species"]["fields"][field] = self.ontology["entities"]["species"]["fields"][field]
                    
            # Add species to target entities to ensure connection
            target_entities.add("species")
        
        # Find paths between target entities using BFS
        relationship_graph = {}
        for rel in self.ontology["relationships"]:
            if rel["from"] not in relationship_graph:
                relationship_graph[rel["from"]] = []
            relationship_graph[rel["from"]].append(rel)
        
        # Track processed entity pairs to avoid duplicates
        processed_pairs = set()
        
        # For each pair of target entities, find shortest path
        entities_list = list(target_entities)
        for i in range(len(entities_list)):
            for j in range(i + 1, len(entities_list)):
                start_entity = entities_list[i]
                end_entity = entities_list[j]
                
                # Skip if we've already processed this pair
                pair_key = tuple(sorted([start_entity, end_entity]))
                if pair_key in processed_pairs:
                    continue
                
                processed_pairs.add(pair_key)
                
                # BFS to find shortest path
                queue = [(start_entity, [])]
                visited = {start_entity}
                
                while queue:
                    current, path = queue.pop(0)
                    
                    if current == end_entity:
                        # Found path - add the relationships
                        for rel in path:
                            # Check for duplicate or reverse relationships
                            duplicate = False
                            for existing_edge in result["edges"]:
                                if (rel["from"] == existing_edge["from"] and rel["to"] == existing_edge["to"]) or \
                                (rel["from"] == existing_edge["to"] and rel["to"] == existing_edge["from"]):
                                    duplicate = True
                                    break
                            
                            if not duplicate:
                                result["edges"].append(rel)
                        break
                    
                    # Try relationships from current entity
                    if current in relationship_graph:
                        for rel in relationship_graph[current]:
                            next_entity = rel["to"]
                            if next_entity not in visited:
                                visited.add(next_entity)
                                queue.append((next_entity, path + [rel]))
                    
                    # Check reverse relationships too
                    for entity, rels in relationship_graph.items():
                        for rel in rels:
                            if rel["to"] == current and entity not in visited:
                                # Create reversed relationship
                                reversed_rel = {
                                    "from": rel["to"],
                                    "to": rel["from"],
                                    "path": rel["path"],
                                    "sql": rel["sql"],
                                    "cardinality": "one-to-many" if rel["cardinality"] == "many-to-one" else 
                                                "many-to-one" if rel["cardinality"] == "one-to-many" else
                                                rel["cardinality"],
                                    "description": rel.get("description", "")
                                }
                                visited.add(entity)
                                queue.append((entity, path + [reversed_rel]))
        
        # Now add intermediate entities and relationship fields
        for edge in result["edges"]:
            path_elements = edge["path"].split(" -> ")
            for element in path_elements:
                entity, field = element.split(".")
                
                # Add intermediate entity if needed
                if entity not in result["nodes"] and entity in self.ontology["entities"]:
                    result["nodes"][entity] = {
                        "table": self.ontology["entities"][entity]["table"],
                        "description": self.ontology["entities"][entity].get("description", ""),
                        "fields": {}
                    }
                
                # Add field needed for the join relationship
                if entity in result["nodes"] and field not in result["nodes"][entity]["fields"]:
                    if entity in self.ontology["entities"] and field in self.ontology["entities"][entity]["fields"]:
                        result["nodes"][entity]["fields"][field] = self.ontology["entities"][entity]["fields"][field]
        
        # If we still don't have a relationship between trna and species, add it manually
        trna_to_species_found = False
        for edge in result["edges"]:
            if (edge["from"] == "trna" and edge["to"] == "species") or (edge["from"] == "species" and edge["to"] == "trna"):
                trna_to_species_found = True
                break
        
        if not trna_to_species_found and "trna" in result["nodes"] and "species" in result["nodes"]:
            # Find the relationship in the ontology
            trna_species_rel = None
            for rel in self.ontology["relationships"]:
                if rel["from"] == "trna" and rel["to"] == "species":
                    trna_species_rel = rel
                    break
                elif rel["from"] == "species" and rel["to"] == "trna":
                    # Create reversed relationship
                    trna_species_rel = {
                        "from": "trna",
                        "to": "species",
                        "path": rel["path"].replace(f"{rel['from']}.{rel['path'].split('.')[-1]}", f"species.id"),
                        "sql": rel["sql"].replace("LEFT JOIN", "LEFT JOIN").replace(f"{rel['from']} AS", "species AS"),
                        "cardinality": "many-to-one" if rel["cardinality"] == "one-to-many" else "one-to-many",
                        "description": f"Link from tRNA to its species"
                    }
                    break
            
            # If we found a relationship, add it
            if trna_species_rel:
                result["edges"].append(trna_species_rel)
            else:
                # Create a default relationship if none found
                default_rel = {
                    "from": "trna",
                    "to": "species",
                    "path": "trna.species_id -> species.id",
                    "sql": "LEFT JOIN species AS s ON t.species_id = s.id",
                    "cardinality": "many-to-one",
                    "description": "Link from tRNA to its species"
                }
                result["edges"].append(default_rel)
                
                # Make sure the fields used in the relationship are included
                if "species_id" not in result["nodes"]["trna"]["fields"] and "species_id" in self.ontology["entities"]["trna"]["fields"]:
                    result["nodes"]["trna"]["fields"]["species_id"] = self.ontology["entities"]["trna"]["fields"]["species_id"]
                
                if "id" not in result["nodes"]["species"]["fields"] and "id" in self.ontology["entities"]["species"]["fields"]:
                    result["nodes"]["species"]["fields"]["id"] = self.ontology["entities"]["species"]["fields"]["id"]
        
        return result

    def format_subgraph(self, subgraph: Dict[str, Any]) -> str:
        """
        Format the subgraph into a human-readable string.
        
        Args:
            subgraph: The subgraph dictionary
            
        Returns:
            Formatted string representation
        """
        output = []
        output.append("=== SUBGRAPH EXTRACTED FROM ONTOLOGY ===\n")
        
        # Include the ontology description at the top
        if "description" in self.ontology:
            output.append("ONTOLOGY DESCRIPTION:")
            output.append(self.ontology["description"])
            output.append("")  # Empty line for spacing

        # Format entities and fields
        output.append("ENTITIES:")
        for entity, data in subgraph["nodes"].items():
            output.append(f"\n--- {entity} ({data['table']}) ---")
            output.append(data.get("description", ""))
            output.append("\nFields:")
            
            for field, properties in data["fields"].items():
                output.append(f"  • {field}: {properties['type']}")
                output.append(f"    {properties.get('description', '')}")
                if "example" in properties:
                    output.append(f"    Example: {properties['example']}")
        
        # Format relationships
        output.append("\n\nRELATIONSHIPS:")
        for edge in subgraph["edges"]:
            output.append(f"\n--- {edge['from']} → {edge['to']} ({edge['cardinality']}) ---")
            output.append(edge.get("description", ""))
            output.append(f"Path: {edge['path']}")
            output.append(f"SQL: {edge['sql']}")
        
        return "\n".join(output)

    def generate_sql_hint(self, subgraph: Dict[str, Any]) -> str:
        """
        Generate SQL query hint from the subgraph.
        
        Args:
            subgraph: The subgraph dictionary
            
        Returns:
            SQL query hint as a string
        """
        if not subgraph["edges"]:
            tables = [data["table"] for data in subgraph["nodes"].values()]
            return f"-- Suggested tables: {', '.join(tables)}"
        
        # Find a starting entity - prefer tRNA as the central entity
        start_entity = "trna" if "trna" in subgraph["nodes"] else next(iter(subgraph["nodes"]))
        
        # Generate SQL JOIN
        start_table = subgraph["nodes"][start_entity]["table"]
        sql_parts = [f"FROM {start_table} AS {start_table[0]}"]
        visited = {start_entity}
        
        # Track added joins to avoid duplicates
        added_joins = set()
        
        # Build a relationship graph
        relationship_graph = {}
        for edge in subgraph["edges"]:
            if edge["from"] not in relationship_graph:
                relationship_graph[edge["from"]] = []
            relationship_graph[edge["from"]].append(edge)
        
        # BFS to traverse and join all related entities
        queue = [start_entity]
        while queue:
            current = queue.pop(0)
            
            # Process outgoing relationships
            if current in relationship_graph:
                for edge in relationship_graph[current]:
                    to_entity = edge["to"]
                    if to_entity not in visited:
                        visited.add(to_entity)
                        queue.append(to_entity)
                        
                        # Generate a key for this join to avoid duplicates
                        join_key = f"{edge['from']}->{edge['to']}"
                        if join_key not in added_joins:
                            added_joins.add(join_key)
                            sql_parts.append(edge["sql"])
        
        return "\n".join(sql_parts)
        
    async def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute a SQL query against the tRNA database.
        
        Args:
            sql_query: The SQL query to execute
            
        Returns:
            Dictionary with query results and metadata
        """
        start_time = time.time()
        results = []
        column_names = []
        error_message = None
        
        try:
            # Check if DB file exists
            if not os.path.exists(DB_PATH):
                raise FileNotFoundError(f"Database file not found: {DB_PATH}")
                
            logging.info(f"Connecting to database: {DB_PATH}")
            logging.info(f"Executing SQL query: {sql_query}")
            
            # Execute the query
            async with aiosqlite.connect(DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(sql_query)
                rows = await cursor.fetchall()
                
                # Get column names
                column_names = [description[0] for description in cursor.description]
                
                # Convert rows to dictionaries
                for row in rows:
                    result_dict = {column: row[column] for column in column_names}
                    results.append(result_dict)
                
                await cursor.close()
                
            logging.info(f"Query executed successfully, returned {len(results)} rows")
        
        except Exception as e:
            error_message = str(e)
            logging.error(f"Error executing SQL query: {error_message}")
            logging.error(f"SQL: {sql_query}")
            logging.error(f"DB_PATH: {DB_PATH}")
            # Print current directory for debugging
            logging.error(f"Current directory: {os.getcwd()}")
        
        finally:
            execution_time = time.time() - start_time
        
        # Return execution results
        result_obj = {
            "success": error_message is None,
            "error": error_message,
            "column_names": column_names,
            "results": results,
            "row_count": len(results),
            "execution_time": execution_time,
            "sql": sql_query
        }
        
        # Log summary of result
        if error_message:
            logging.error(f"Query failed: {error_message}")
        else:
            logging.info(f"Query returned {len(results)} rows in {execution_time:.2f}s")
            
        return result_obj

    async def query_llm(self,
                       model_name: str,
                       prompt: str,
                       include_ontology: bool = True) -> Dict[str, Any]:
        """
        Query an LLM model with the given prompt.
        Uses direct access to the LiteLLM proxy or Ollama API depending on the model name.

        Args:
            model_name: Name of the LLM model to use
            prompt: The prompt to send to the model
            include_ontology: Whether to include the ontology subgraph in the prompt

        Returns:
            Response from the LLM with extracted components
        """
        start_time = time.time()

        try:
            # Check if this is an Ollama model
            is_ollama = model_name.startswith("ollama:")
            actual_model = model_name.replace("ollama:", "") if is_ollama else model_name

            if is_ollama:
                # Use Ollama API
                logging.info(f"Using Ollama API for model: {actual_model}")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": actual_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.2
                            }
                        },
                        timeout=120.0  # Longer timeout for Ollama
                    )

                    result = response.json()
                    full_response = result.get("response", "")

                    # Log full response for debugging
                    logging.info(f"Ollama API response received, length: {len(full_response)}")
                    logging.info(f"Ollama full response: {full_response}")
            else:
                # Direct access to LiteLLM proxy (existing behavior)
                logging.info(f"Using LiteLLM proxy for model: {model_name}")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{LITELLM_BASE_URL}/v1/completions",
                        json={
                            "model": model_name,
                            "prompt": prompt,
                            "max_tokens": 2000,
                            "temperature": 0.2
                        },
                        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                        timeout=60.0
                    )

                    result = response.json()
                    full_response = result.get("choices", [{}])[0].get("text", "")

                    # Log full response for debugging (same as with Ollama)
                    logging.info(f"LiteLLM API response received, length: {len(full_response)}")
                    logging.info(f"LiteLLM full response: {full_response}")

        except Exception as e:
            logging.error(f"Error querying model {model_name}: {str(e)}")
            full_response = f"Error: {str(e)}"

        finally:
            latency = time.time() - start_time

        # Create response object
        response = {
            "model": model_name,
            "full_response": full_response,
            "latency": latency,
            "include_ontology": include_ontology
        }
        
        # Extract SQL and comments
        sql_blocks = []
        comments = []
        
        # First, collect all segments (text before SQL, SQL blocks, text after SQL)
        segments = []
        current_segment = ""
        in_sql_block = False
        lines = full_response.split("\n")
        
        for i, line in enumerate(lines):
            if line.startswith("```sql"):
                # End of a non-SQL segment, start of an SQL block
                if current_segment.strip():
                    segments.append({"type": "comment", "content": current_segment.strip()})
                in_sql_block = True
                current_segment = ""
            elif line.startswith("```") and in_sql_block:
                # End of an SQL block
                segments.append({"type": "sql", "content": current_segment.strip()})
                in_sql_block = False
                current_segment = ""
            else:
                # Content for the current segment
                if in_sql_block and not line.startswith("```sql"):
                    current_segment += line + "\n"
                elif not in_sql_block:
                    current_segment += line + "\n"
        
        # Add the last segment if it exists
        if current_segment.strip():
            segments.append({"type": "comment", "content": current_segment.strip()})
        
        # Extract SQL blocks and collect all comment segments
        full_comments = ""
        for segment in segments:
            if segment["type"] == "sql":
                sql_blocks.append(segment["content"])
            else:
                full_comments += segment["content"] + "\n\n"
        
        # Store the full comments (trim extra newlines)
        if full_comments.strip():
            comments.append(full_comments.strip())
            
        # Add extracted components to the response
        response["sql_blocks"] = sql_blocks
        response["comments"] = comments
        
        return response

    async def query_with_params(self, 
                         query_text: str,
                         model_name: str,
                         include_ontology: bool = True,
                         execute_query: bool = True,
                         k: int = 5) -> Dict[str, Any]:
        """
        Process a single query with a specific model and ontology setting.
        Streamlined version for single model/single trial use case.
        
        Args:
            query_text: The natural language query
            model_name: Name of the LLM model to use
            include_ontology: Whether to include ontology subgraph in the prompt
            execute_query: Whether to execute the generated SQL query
            k: Number of fields to retrieve
            
        Returns:
            Dictionary with query results
        """
        # Get the timestamp
        timestamp = datetime.now().isoformat()
        start_time = time.time()
        
        # Step 1: Retrieve top-k semantically similar fields
        logging.info(f"Retrieving top-{k} fields for query: {query_text}")
        top_fields = await self.retrieve_top_k_fields(query_text, k)
        
        # Step 2: Extract detailed information for each field
        logging.info("Extracting field details")
        field_details = [self.extract_field_info(field_id) for field_id, _ in top_fields]
        
        # Step 3: Extract a minimal subgraph connecting these fields
        logging.info("Extracting minimal subgraph")
        target_field_ids = [field_id for field_id, _ in top_fields]
        subgraph = self.extract_minimal_subgraph(target_field_ids)
        
        # Step 4: Format the subgraph for human readability
        logging.info("Formatting subgraph")
        formatted_subgraph = self.format_subgraph(subgraph)
        
        # Step 5: Generate SQL hint
        logging.info("Generating SQL hint")
        sql_hint = self.generate_sql_hint(subgraph)
        
        # No need to extract ontology description separately as it's included in the formatted subgraph
        
        # Preparation phase latency
        prep_latency = time.time() - start_time
        logging.info(f"Preparation completed in {prep_latency:.2f} seconds")
        
        # Step 7: Query the model
        # Prepare the prompt
        prompt = f"Query: {query_text}\n\n"
        
        if include_ontology:
            prompt += f"{formatted_subgraph}\n\n"
            
        prompt += f"SQL Hint:\n{sql_hint}\n\n"
        prompt += "Please provide a SQL query to answer this question and explain your approach. Format the SQL query inside ```sql``` blocks."
        
        # Query the model
        logging.info(f"Querying model {model_name}")
        model_response = await self.query_llm(model_name, prompt, include_ontology)
        
        # Step 7: Execute the SQL query if requested
        query_results = None
        if execute_query and model_response["sql_blocks"]:
            logging.info("Executing SQL query")
            sql_query = model_response["sql_blocks"][0]  # Use the first SQL block
            query_results = await self.execute_sql(sql_query)
        
        # Total processing latency
        total_latency = time.time() - start_time
        logging.info(f"Total processing completed in {total_latency:.2f} seconds")
        
        # Return comprehensive results with removed redundancy
        return {
            "query": query_text,
            "timestamp": timestamp,
            "latency": {
                "preparation": prep_latency,
                "total": total_latency
            },
            "field_retrieval": {
                "top_fields": [{"field_id": field_id, "similarity": similarity} for field_id, similarity in top_fields],
                "field_details": field_details
            },
            "subgraph": subgraph,
            "formatted_subgraph": formatted_subgraph,
            "sql_hint": sql_hint,
            "model_response": model_response,
            "query_results": query_results
        }

    async def process_query_with_models(self, 
                                     query_text: str, 
                                     models: List[Dict[str, Any]], 
                                     execute_query: bool = True,
                                     k: int = 5) -> Dict[str, Any]:
        """
        Process a natural language query with multiple LLM models and trial configurations.
        
        Args:
            query_text: The natural language query
            models: List of model configurations, each with name and trial options
            execute_query: Whether to execute the generated SQL queries
            k: Number of fields to retrieve
            
        Returns:
            Dictionary with comprehensive query results
        """
        # Get the timestamp
        timestamp = datetime.now().isoformat()
        start_time = time.time()
        
        # Step 1: Retrieve top-k semantically similar fields
        logging.info(f"Retrieving top-{k} fields for query: {query_text}")
        top_fields = await self.retrieve_top_k_fields(query_text, k)
        
        # Step 2: Extract detailed information for each field
        logging.info("Extracting field details")
        field_details = [self.extract_field_info(field_id) for field_id, _ in top_fields]
        
        # Step 3: Extract a minimal subgraph connecting these fields
        logging.info("Extracting minimal subgraph")
        target_field_ids = [field_id for field_id, _ in top_fields]
        subgraph = self.extract_minimal_subgraph(target_field_ids)
        
        # Step 4: Format the subgraph for human readability
        logging.info("Formatting subgraph")
        formatted_subgraph = self.format_subgraph(subgraph)
        
        # Step 5: Generate SQL hint
        logging.info("Generating SQL hint")
        sql_hint = self.generate_sql_hint(subgraph)
        
        # No need to extract ontology description separately as it's included in the formatted subgraph
        
        # Preparation phase latency
        prep_latency = time.time() - start_time
        logging.info(f"Preparation completed in {prep_latency:.2f} seconds")
        
        # Step 7: Query each model with each trial configuration
        model_results = {}
        
        # Create tasks for all model queries
        tasks = []
        model_trial_map = []  # To keep track of which task corresponds to which model/trial
        
        for model_config in models:
            model_name = model_config["name"]
            trials = model_config.get("trials", [{"name": "default"}])
            
            model_results[model_name] = {}
            
            for trial in trials:
                trial_name = trial.get("name", "default")
                include_ontology = trial.get("include_ontology", True)
                
                # Prepare the prompt
                prompt = f"Query: {query_text}\n\n"
                
                if include_ontology:
                    prompt += f"{formatted_subgraph}\n\n"
                    
                prompt += f"SQL Hint:\n{sql_hint}\n\n"
                prompt += "Please provide a SQL query to answer this question and explain your approach. Format the SQL query inside ```sql``` blocks."
                
                # Create a task for querying this model with this trial
                tasks.append(self.query_llm(model_name, prompt, include_ontology))
                model_trial_map.append((model_name, trial_name))
        
        # Run all queries in parallel
        logging.info(f"Querying {len(tasks)} model/trial combinations in parallel")
        model_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results by model and trial
        for i, response in enumerate(model_responses):
            model_name, trial_name = model_trial_map[i]
            
            if isinstance(response, Exception):
                # Handle exceptions
                logging.error(f"Error querying {model_name} ({trial_name}): {str(response)}")
                model_results[model_name][trial_name] = {
                    "model": model_name,
                    "error": str(response),
                    "latency": 0,
                    "full_response": f"Error: {str(response)}",
                    "sql_blocks": [],
                    "comments": [],
                    "query_results": None
                }
            else:
                # Execute SQL if requested and available
                query_results = None
                if execute_query and response["sql_blocks"]:
                    logging.info(f"Executing SQL from {model_name} ({trial_name})")
                    sql_query = response["sql_blocks"][0]  # Use the first SQL block
                    query_results = await self.execute_sql(sql_query)
                    logging.info(f"SQL execution completed for {model_name} ({trial_name}) with {query_results['row_count']} rows")
                
                # Add query results to response
                response_with_results = dict(response)
                response_with_results["query_results"] = query_results
                model_results[model_name][trial_name] = response_with_results
        
        # Total processing latency
        total_latency = time.time() - start_time
        logging.info(f"Total processing completed in {total_latency:.2f} seconds")
        
        # Return comprehensive results with removed redundancy
        return {
            "query": query_text,
            "timestamp": timestamp,
            "latency": {
                "preparation": prep_latency,
                "total": total_latency
            },
            "field_retrieval": {
                "top_fields": [{"field_id": field_id, "similarity": similarity} for field_id, similarity in top_fields],
                "field_details": field_details
            },
            "subgraph": subgraph,
            "formatted_subgraph": formatted_subgraph,
            "sql_hint": sql_hint,
            "model_results": model_results
        }

async def query_api(
    query_text: str,
    models: List[str] = None,  
    trials: List[Dict[str, Any]] = None,
    execute_query: bool = True,
    k: int = 5,
    embedding_file: str = "trna_index_v01.json",
    ontology_file: str = "trnadb_ontology_v01.json",
    db_path: str = None
) -> Dict[str, Any]:
    """
    Main API function to process a tRNA query.
    
    Args:
        query_text: The natural language query
        models: List of model names to query (if None, uses "openai-gpt-4")
        trials: List of trial configurations with include_ontology setting (if None, uses a single default trial)
        execute_query: Whether to execute the generated SQL query
        k: Number of fields to retrieve
        embedding_file: Path to embedding file
        ontology_file: Path to ontology file
        db_path: Path to the SQLite database (if None, uses default DB_PATH)
        
    Returns:
        Query results as a JSON object
    """
    # Configure logging with a file handler to ensure logs are saved
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trna_query.log')
        ],
        force=True  # Force reconfiguration to ensure our handlers are used
    )
    
    logging.info(f"Processing query: '{query_text}' with models: {models}")
    logging.info(f"Execute query: {execute_query}, top-k: {k}")
    
    # Use default model if none specified
    if not models:
        models = ["openai-gpt-4"]
    
    # Use default trial if none specified
    if not trials:
        trials = [{"name": "default", "include_ontology": True}]
    
    # Override DB_PATH if specified
    if db_path:
        global DB_PATH
        original_db_path = DB_PATH
        DB_PATH = db_path
        logging.info(f"Overriding default DB path: {original_db_path} -> {DB_PATH}")
    
    # Check if necessary files exist
    for file_path, file_type in [(embedding_file, "embedding file"), (ontology_file, "ontology file"), (DB_PATH, "database file")]:
        if not os.path.exists(file_path):
            error_msg = f"{file_type.capitalize()} not found: {file_path}"
            logging.error(error_msg)
            return {
                "error": error_msg,
                "query": query_text,
                "success": False
            }
    
    try:
        # Initialize the retriever
        logging.info(f"Initializing retriever with: {embedding_file}, {ontology_file}")
        retriever = AsyncTRNARetriever(
            embedding_file=embedding_file,
            ontology_file=ontology_file
        )
        
        # Process the query with specified parameters
        if len(models) == 1 and len(trials) == 1:
            # Simple case - single model, single trial
            logging.info(f"Processing with single model: {models[0]}")
            results = await retriever.query_with_params(
                query_text=query_text,
                model_name=models[0],
                include_ontology=trials[0].get("include_ontology", True),
                execute_query=execute_query,
                k=k
            )
        else:
            # Multiple models or trials
            logging.info(f"Processing with multiple models/trials: {models}")
            model_configs = []
            for model_name in models:
                model_configs.append({
                    "name": model_name,
                    "trials": trials
                })
            
            results = await retriever.process_query_with_models(
                query_text=query_text, 
                models=model_configs, 
                execute_query=execute_query,
                k=k
            )
        
        # Log successful completion
        logging.info(f"Query processing completed successfully")
        return results
        
    except Exception as e:
        # Log any exceptions that occur during processing
        error_msg = f"Error in query_api: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return {
            "error": error_msg,
            "query": query_text,
            "success": False
        }





# Example usage
async def main():
    # Simple query with a single model and SQL execution
    query = "Find all tRNAs with methionine anticodons"
    results = await query_api(
        query_text=query, 
        models=["openai-gpt-4"],
        execute_query=True
    )
    
    # Print query results if available
    if results.get("query_results"):
        print(f"Database query returned {results['query_results']['row_count']} rows")
        # Print first few results
        for i, row in enumerate(results["query_results"]["results"][:3]):
            print(f"Row {i+1}: {row}")
            if i >= 2 and len(results["query_results"]["results"]) > 3:
                print(f"...and {len(results['query_results']['results']) - 3} more rows")
                break
    elif results.get("model_response") and results["model_response"].get("query_results"):
        query_results = results["model_response"]["query_results"]
        print(f"Database query returned {query_results['row_count']} rows")
        # Print first few results
        for i, row in enumerate(query_results["results"][:3]):
            print(f"Row {i+1}: {row}")
            if i >= 2 and len(query_results["results"]) > 3:
                print(f"...and {len(query_results['results']) - 3} more rows")
                break
    else:
        print(json.dumps(results, indent=2))
        
    # Print subgraph info
    if results.get("subgraph"):
        print("\nSubgraph extracted with:")
        print(f"- {len(results['subgraph']['nodes'])} nodes (entities)")
        print(f"- {len(results['subgraph']['edges'])} edges (relationships)")
    
    # Query with specific trials and multiple models
    custom_trials = [
        {"name": "with_ontology", "include_ontology": True},
        {"name": "without_ontology", "include_ontology": False}
    ]
    
    # Query with multiple models and custom trials
    query2 = "Get tRNAs for SeC isotype"
    results2 = await query_api(
        query_text=query2,
        models=["openai-gpt-4", "Ministral-8b"],
        trials=custom_trials,
        execute_query=True
    )
    
    # Print summary of results
    print(f"\nMulti-model query completed with {len(results2['model_results'])} models")
    for model_name, trials in results2["model_results"].items():
        for trial_name, response in trials.items():
            query_results = response.get("query_results")
            if query_results and query_results["success"]:
                print(f"{model_name} ({trial_name}): Query returned {query_results['row_count']} rows")
            else:
                print(f"{model_name} ({trial_name}): Query execution failed or not performed")
    
    # Print subgraph info for second query
    if results2.get("subgraph"):
        print("\nSecond query subgraph extracted with:")
        print(f"- {len(results2['subgraph']['nodes'])} nodes (entities)")
        print(f"- {len(results2['subgraph']['edges'])} edges (relationships)")
        
        # Print the actual formatted subgraph to verify content
        print("\nFormatted subgraph sample (first 200 chars):")
        print(results2.get("formatted_subgraph", "")[:200] + "...")
if __name__ == "__main__":
    asyncio.run(main())
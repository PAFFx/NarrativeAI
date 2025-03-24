import logging
from typing import List, Optional, Dict, Any, Union
import urllib.parse
import uuid
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from ...utils import format_rfc3339_datetime

# Load environment variables
load_dotenv()

# Import Weaviate v4 client with explicit instantiation classes
import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.init import AdditionalConfig, Timeout, Auth
from weaviate.classes.config import Configure, Vectorizers, DataType
from weaviate.classes.query import Filter, MetadataQuery, Rerank
from weaviate.exceptions import WeaviateQueryError
        

logger = logging.getLogger(__name__)

class WeaviateClient:
    """Singleton client for Weaviate vector database interactions."""
    
    # Singleton instance
    _instance = None
    
    def __new__(cls, connection_url: str = None, api_key: str = None):
        """Implement singleton pattern."""
        if cls._instance is None:
            logger.info("Creating new WeaviateClient singleton instance")
            cls._instance = super(WeaviateClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, connection_url: str = None, api_key: str = None):
        """Initialize a Weaviate client.
        
        Args:
            connection_url: URL to connect to Weaviate.
                            If None, uses WEAVIATE_URL environment variable.
            api_key: API key for authentication.
                     If None, uses WEAVIATE_API_KEY environment variable.
        """
        # Skip initialization if already initialized
        if self._initialized:
            return
            
        self.connection_url = connection_url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        parsed_url = urllib.parse.urlparse(self.connection_url)
        
        # Determine if using secure connection
        is_secure = parsed_url.scheme == "https"
        
        # Extract host and port
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or (443 if is_secure else 80)
        
        # Set up auth if API key is provided
        auth_config = Auth.api_key(self.api_key) if self.api_key else None
        
        # Create connection parameters
        connection_params = ConnectionParams.from_params(
            http_host=host,
            http_port=port,
            http_secure=is_secure,
            grpc_host=host,
            grpc_port=50051,  # Default gRPC port
            grpc_secure=is_secure,
        )
        
        # Initialize client with explicit instantiation
        self.client = weaviate.WeaviateClient(
            connection_params=connection_params,
            auth_client_secret=auth_config,
            additional_headers={
                "X-OpenAI-Api-Key": self.openai_api_key  # For OpenAI modules if used
            },
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120),  # Values in seconds
            ),
            skip_init_checks=False
        )
        
        # Explicitly connect to the server
        self.client.connect()
        
        logger.info(f"Successfully connected to Weaviate at {self.connection_url}")
        
        # Check connection
        if not self.client.is_ready():
            logger.warning("Weaviate client initialized but server not ready")
            raise ConnectionError("Weaviate server not ready")
        
        self._initialized = True
    
    @classmethod
    def close(cls):
        """Close the singleton instance."""
        if cls._instance is not None:
            try:
                if hasattr(cls._instance, 'client'):
                    cls._instance.client.close()
                    logger.info("Closed Weaviate client connection")
            except Exception as e:
                logger.warning(f"Error closing Weaviate client: {str(e)}")
            cls._instance = None
            logger.info("WeaviateClient singleton instance reset")
            
    
    def _ensure_schema_exists(self, collection_name: str = "StoryMemory"):
        """Ensure the required schema exists in Weaviate.
        
        Args:
            collection_name: Name of the collection to check/create
            
        Returns:
            Collection object if successful, None otherwise
        """
        try:
            # Check if the collection exists by attempting to get it directly
            if self.client.collections.exists(collection_name):
                # Try to get the collection - if it exists, this will succeed
                collection = self.client.collections.get(collection_name)
                return collection
            else:
                # Create a basic collection with minimal configuration
                logger.info(f"Creating new collection: {collection_name}")
                collection = self.client.collections.create(
                    name=collection_name,
                    description="Memory elements from the interactive story",
                    vectorizer_config=Configure.Vectorizer.text2vec_openai(
                        model="ada",
                        model_version="002", 
                        type_="text"
                    ),
                    properties=[
                        {
                            "name": "text",
                            "description": "The text content of the memory",
                            "data_type": DataType.TEXT,
                            "vectorize": True,
                        },
                        {
                            "name": "characters",
                            "description": "Characters involved in the memory",
                            "data_type": DataType.TEXT_ARRAY,
                            "vectorize": True,
                        },
                        {
                            "name": "locations",
                            "description": "Locations where the memory takes place",
                            "data_type": DataType.TEXT_ARRAY,
                            "vectorize": True,
                        },
                        {
                            "name": "themes",
                            "description": "Themes present in the memory",
                            "data_type": DataType.TEXT_ARRAY,
                            "vectorize": True,
                        },
                        {
                            "name": "timestamp",
                            "description": "When the memory occurred",
                            "data_type": DataType.DATE,
                        },
                        {
                            "name": "chapter",
                            "description": "Chapter of the story",
                            "data_type": DataType.TEXT,
                        },
                        {
                            "name": "importance_score",
                            "description": "Importance of the memory (1-10)",
                            "data_type": DataType.NUMBER,
                        },
                        {
                            "name": "storyline",
                            "description": "Which storyline this memory belongs to",
                            "data_type": DataType.TEXT,
                        },
                        {
                            "name": "emotion",
                            "description": "Primary emotion in the memory",
                            "data_type": DataType.TEXT,
                        }
                    ]
                )
                
                logger.info(f"Created {collection_name} collection successfully")
                return collection
            
        except Exception as e:
            logger.exception(f"Error accessing or creating collection: {str(e)}")
            return None
    
    def query(self, collection_name: str, query_params: dict) -> List[dict]:
        """Query the Weaviate database for relevant memories.
        
        Args:
            collection_name: Name of the collection to query
            query_params: Query parameters
            
        Returns:
            List of memories based on query parameters
            
        Raises:
            Exception: If query fails or no results found
        """
        # Get the collection
        collection = self._ensure_schema_exists(collection_name)
        # Check if collection is None (not if it evaluates to False)
        if collection is None:
            raise ValueError(f"Could not get or create collection: {collection_name}")
        
        # Extract parameters
        query_text = query_params.get("query_text", "story elements")
        limit = query_params.get("limit", 5)
        
        # Start building the query 
        query_builder = collection.query

        # Prepare filter if needed
        filter_conditions = []
        
        # Add character filter if provided
        if query_params.get("characters"):
            chars = query_params["characters"]
            if isinstance(chars, list) and len(chars) > 0:
                for char in chars:
                    filter_conditions.append(
                        Filter.by_property("characters").contains_any([char])
                    )
        
        # Add location filter if provided
        if query_params.get("locations"):
            locs = query_params["locations"]
            if isinstance(locs, list) and len(locs) > 0:
                for loc in locs:
                    filter_conditions.append(
                        Filter.by_property("locations").contains_any([loc])
                    )
        
        # Add theme filter if provided
        if query_params.get("themes"):
            themes = query_params["themes"]
            if isinstance(themes, list) and len(themes) > 0:
                for theme in themes:
                    filter_conditions.append(
                        Filter.by_property("themes").contains_any([theme])
                    )
        
        # Add importance score filter if provided
        if query_params.get("importance_score_min"):
            imp_score = float(query_params["importance_score_min"])
            filter_conditions.append(
                Filter.by_property("importance_score").greater_or_equal(imp_score)
            )
        
        # Add storyline filter if provided
        if query_params.get("storyline"):
            storyline = query_params["storyline"]
            filter_conditions.append(
                Filter.by_property("storyline").equal(storyline)
            )
        
        # Combine filters if we have any with OR between conditions
        filter_query = None
        if filter_conditions:
            if len(filter_conditions) == 1:
                filter_query = filter_conditions[0]
            else:
                filter_query = Filter.any_of(filter_conditions)
        
        try:
            # First step: perform hybrid search to get initial results
            # Return a larger initial result set to have more options for reranking
            initial_limit = min(limit * 3, 20)  # Get more results than needed but cap at 20
            
            response = query_builder.hybrid(
                query=query_text,
                limit=initial_limit,
                filters=filter_query,
                return_metadata=MetadataQuery(score=True, distance=True)
            )
            
            if not response.objects:
                # No memories found, return empty list
                return []
            
            logger.info(f"Retrieved {len(response.objects)} initial memories from Weaviate for reranking")
            
            # Second step: rerank the results based on a combination of relevance and importance score
            memories = []
            
            # Process and rerank results
            for obj in response.objects:
                memory = obj.properties
                
                # Add metadata scores
                memory["_relevance_score"] = obj.metadata.score if hasattr(obj.metadata, "score") else 0.5
                memory["_distance"] = obj.metadata.distance if hasattr(obj.metadata, "distance") else 0.5
                
                # Ensure importance_score exists with a valid value
                if "importance_score" not in memory or not isinstance(memory["importance_score"], (int, float)):
                    memory["importance_score"] = 5  # Default medium importance
                
                # Calculate combined score: 0.7 * relevance + 0.3 * normalized importance
                # This weights relevance higher but still considers importance
                normalized_importance = memory["importance_score"] / 10.0  # Convert 1-10 to 0-1 scale
                combined_score = (0.7 * memory["_relevance_score"]) + (0.3 * normalized_importance)
                
                memory["_combined_score"] = combined_score
                memories.append(memory)
            
            # Sort by combined score (descending)
            memories.sort(key=lambda x: x["_combined_score"], reverse=True)
            
            # Limit to requested number
            memories = memories[:limit]
            
            # Clean up internal scoring fields
            for memory in memories:
                if "_relevance_score" in memory:
                    del memory["_relevance_score"]
                if "_distance" in memory:
                    del memory["_distance"]
                if "_combined_score" in memory:
                    del memory["_combined_score"]
            
            logger.info(f"Reranked and returning {len(memories)} memories")
            
            if not memories:
                raise ValueError(f"No memories found after reranking for query: {query_params}")
                
            return memories
            
        except Exception as e:
            logger.exception(f"Error executing Weaviate query: {str(e)}")
            raise ValueError(f"Failed to retrieve memories: {str(e)}")
            
    def add_memory(self, collection_name: str, memory_data: dict) -> str:
        """Add a new memory to the Weaviate database.
        
        Args:
            collection_name: Name of the collection to add to
            memory_data: Memory data to add
            
        Returns:
            ID of the created memory
            
        Raises:
            Exception: If adding memory fails
        """
        # Get the collection
        collection = self._ensure_schema_exists(collection_name)
        if collection is None:  # Explicit check for None, not False
            raise ValueError(f"Could not get or create collection: {collection_name}")
            
        # Ensure timestamps are properly formatted
        if "timestamp" not in memory_data:
            memory_data["timestamp"] = format_rfc3339_datetime()
        
        # Generate a UUID based on the text content for deduplication 
        if "text" in memory_data:
            # A simple deterministic UUID generation
            memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_data["text"]))
        else:
            memory_id = str(uuid.uuid4())
            
        # Add the object with the explicit API
        result = collection.data.insert(
            properties=memory_data,
            uuid=memory_id
        )
        
        logger.info(f"Added memory to Weaviate with ID: {memory_id}")
        return memory_id


class MemoryRetriever:
    """Class for retrieving memories from the vector database."""
    
    def __init__(self, client: WeaviateClient = None, thread_id: str = None, collection_prefix: str = "story_"):
        """Initialize the memory retriever.
        
        Args:
            client: Weaviate client (will use singleton if None)
            thread_id: Unique identifier for the conversation thread
            collection_prefix: Prefix for collection names, default is "story_"
        """
        # Use provided client or get the singleton instance
        self.client = client or WeaviateClient()
        self.thread_id = thread_id
        self.collection_prefix = collection_prefix
        self.collection_name = self._get_collection_name()
        logger.info(f"Created MemoryRetriever for collection: {self.collection_name}")
        
    def _get_collection_name(self) -> str:
        """Get the collection name for the current thread.
        
        If no thread_id is provided, uses the default "StoryMemory" collection.
        
        Returns:
            Collection name
        """
        if not self.thread_id:
            logger.warning("No thread_id provided, using default StoryMemory collection")
            return "StoryMemory"  # Default collection for backward compatibility
        
        # Create a collection name from the thread ID
        # Replace dashes with underscores for valid Weaviate collection names
        safe_thread_id = str(self.thread_id).replace("-", "_")
        collection_name = f"{self.collection_prefix}{safe_thread_id}"
        logger.info(f"Created thread-specific collection name: {collection_name}")
        return collection_name
    
    def set_thread_id(self, thread_id: str) -> None:
        """Update the thread ID and collection name.
        
        Args:
            thread_id: New thread ID to use
        """
        if not thread_id:
            logger.warning("Attempted to set empty thread_id")
            return
            
        logger.info(f"Setting thread ID: {thread_id}")
        
        # If thread_id is the same, no need to update
        if self.thread_id == thread_id:
            logger.info(f"Thread ID already set to {thread_id}, using collection: {self.collection_name}")
            return
            
        # Update thread_id and collection_name
        self.thread_id = thread_id
        old_collection = self.collection_name
        self.collection_name = self._get_collection_name()
        
        # Initialize the collection to make sure it exists
        try:
            self.client._ensure_schema_exists(self.collection_name)
            logger.info(f"Memory retriever switched from '{old_collection}' to '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error initializing collection {self.collection_name}: {str(e)}")
    
    def parse_memory_query(self, query_str: str) -> dict:
        """Parse the memory agent's output into a structured query.
        
        Args:
            query_str: The string output from the memory agent
            
        Returns:
            Structured query dictionary
        """
        # Default values
        query_dict = {
            "query_text": "",
            "characters": [],
            "locations": [],
            "themes": [],
            "importance_score_min": 3,
            "storyline": "",
            "limit": 5
        }
        
        try:
            # Validate input
            if not query_str or not isinstance(query_str, str):
                logger.warning(f"Invalid query string: {type(query_str).__name__}")
                query_dict["query_text"] = "story continuation"
                return query_dict
                
            # Remove any leading/trailing whitespace and handle different line endings
            clean_str = query_str.strip().replace('\r\n', '\n')
            
            # Use a better approach to parse the string by splitting by lines first
            lines = clean_str.split('\n')
            field_pattern = r'^([a-zA-Z_]+):\s*(.*)$'
            
            for line in lines:
                match = re.match(field_pattern, line.strip())
                if match:
                    field_name = match.group(1).lower()
                    field_value = match.group(2).strip()
                    
                    if field_name == 'query_text':
                        query_dict["query_text"] = field_value
                    elif field_name == 'characters':
                        if field_value and field_value.lower() not in ("none", "[]", ""):
                            cleaned_chars = field_value.strip("[]").split(',')
                            query_dict["characters"] = [c.strip() for c in cleaned_chars if c.strip()]
                    elif field_name == 'locations':
                        if field_value and field_value.lower() not in ("none", "[]", ""):
                            cleaned_locs = field_value.strip("[]").split(',')
                            query_dict["locations"] = [l.strip() for l in cleaned_locs if l.strip()]
                    elif field_name == 'themes':
                        if field_value and field_value.lower() not in ("none", "[]", ""):
                            cleaned_themes = field_value.strip("[]").split(',')
                            query_dict["themes"] = [t.strip() for t in cleaned_themes if t.strip()]
                    elif field_name == 'importance_score_min':
                        try:
                            query_dict["importance_score_min"] = int(field_value)
                        except ValueError:
                            logger.warning(f"Invalid importance score: {field_value}")
                    elif field_name == 'limit':
                        try:
                            query_dict["limit"] = int(field_value)
                        except ValueError:
                            logger.warning(f"Invalid limit: {field_value}")
                    elif field_name == 'storyline':
                        query_dict["storyline"] = field_value
            
        except Exception as e:
            logger.exception(f"Error parsing memory query")
            logger.error(f"Original query string: {query_str}")
            # Ensure we have a minimum valid query even if parsing fails
            if not query_dict["query_text"]:
                query_dict["query_text"] = "story continuation"
        
        # Validate final query - ensure we have a query_text
        if not query_dict["query_text"]:
            query_dict["query_text"] = "story continuation"
            
        logger.info(f"Parsed memory query: {query_dict}")
        return query_dict
    
    def retrieve_memories(self, query_dict: dict) -> List[dict]:
        """Retrieve memories from the vector database for the current thread.
        
        Args:
            query_dict: Structured query dictionary
            
        Returns:
            List of memory objects
            
        Raises:
            Exception: If memory retrieval fails
        """
        # Type checking and validation
        if not isinstance(query_dict, dict):
            raise TypeError(f"Invalid query_dict type: {type(query_dict).__name__}")
            
        # Ensure we have at least a query_text
        if "query_text" not in query_dict or not query_dict["query_text"]:
            logger.warning("Missing query_text in query_dict, using default")
            query_dict["query_text"] = "story continuation"
        
        # Make sure the collection exists before querying
        self.client._ensure_schema_exists(self.collection_name)
        logger.info(f"Retrieving memories from collection: {self.collection_name}")
            
        # Query the database using the thread-specific collection
        memories = self.client.query(self.collection_name, query_dict)
        logger.info(f"Retrieved {len(memories)} memories from collection {self.collection_name}")
        return memories
            
    def add_memory(self, memory_data: dict) -> str:
        """Add a new memory to the thread-specific collection.
        
        Args:
            memory_data: Memory data to add
            
        Returns:
            ID of the created memory
        """
        try:
            # Validate the memory data
            required_fields = ["text", "characters", "locations", "themes"]
            for field in required_fields:
                if field not in memory_data:
                    logger.warning(f"Missing required field '{field}' in memory data")
                    if field == "text":
                        return ""  # Can't add a memory without text
                    elif field in ["characters", "locations", "themes"]:
                        memory_data[field] = []  # Use empty lists as defaults
            
            # Ensure lists are properly formatted
            for list_field in ["characters", "locations", "themes"]:
                if not isinstance(memory_data[list_field], list):
                    # If the field is a string, try to convert it to a list
                    if isinstance(memory_data[list_field], str):
                        memory_data[list_field] = [item.strip() for item in memory_data[list_field].split(",") if item.strip()]
                    else:
                        memory_data[list_field] = []
            
            # Ensure importance_score is a number
            if "importance_score" in memory_data:
                try:
                    # If it's a 0-1 float, convert to 1-10 scale
                    if isinstance(memory_data["importance_score"], float) and memory_data["importance_score"] <= 1.0:
                        memory_data["importance_score"] = int(memory_data["importance_score"] * 10) + 1
                    
                    # Ensure it's between 1 and 10
                    importance = float(memory_data["importance_score"])
                    memory_data["importance_score"] = min(10, max(1, importance))
                except (ValueError, TypeError):
                    memory_data["importance_score"] = 5  # Default to medium importance
            else:
                memory_data["importance_score"] = 5
            
            # Ensure timestamp exists
            if "timestamp" not in memory_data or not memory_data["timestamp"]:
                memory_data["timestamp"] = format_rfc3339_datetime()
            elif isinstance(memory_data["timestamp"], str) and "Z" not in memory_data["timestamp"] and "+" not in memory_data["timestamp"]:
                # If timestamp exists but doesn't have timezone info, format it properly
                try:
                    dt = datetime.fromisoformat(memory_data["timestamp"])
                    memory_data["timestamp"] = format_rfc3339_datetime(dt)
                except ValueError:
                    # If timestamp couldn't be parsed, set a new one
                    memory_data["timestamp"] = format_rfc3339_datetime()
            
            # Make sure the collection exists before adding memory
            # This is critical to ensure we're using the right collection
            self.client._ensure_schema_exists(self.collection_name)
            
            # Add the memory to the thread-specific Weaviate collection
            logger.info(f"Adding memory to collection: {self.collection_name}")
            memory_id = self.client.add_memory(self.collection_name, memory_data)
            logger.info(f"Added new memory with ID: {memory_id} to collection {self.collection_name}")
            return memory_id
        except Exception as e:
            logger.exception(f"Error adding memory to collection {self.collection_name}: {str(e)}")
            return ""


# Initialize the singleton client to use across the application
weaviate_client = WeaviateClient()
logger.info("Initialized WeaviateClient singleton")

# Helper function to create a memory retriever with the singleton client
def create_memory_retriever(thread_id: str = None, collection_prefix: str = "story_") -> MemoryRetriever:
    """Create a new MemoryRetriever using the singleton WeaviateClient.
    
    Args:
        thread_id: Optional thread ID to associate with this retriever
        collection_prefix: Prefix for collection names (default: "story_")
        
    Returns:
        New MemoryRetriever instance
    """
    return MemoryRetriever(weaviate_client, thread_id, collection_prefix)

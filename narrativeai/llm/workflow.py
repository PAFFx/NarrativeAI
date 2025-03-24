import logging
from .states import GraphState
from .agents.writer_agent import WriterAgent
from .agents.memory_agent import MemoryAgent
from .agents.tools.context_vector_db import create_memory_retriever
from .llm import ModelName
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain.schema import AIMessage, SystemMessage, HumanMessage
from typing import List, Dict, Optional, Literal
from .utils import format_rfc3339_datetime
import uuid

logger = logging.getLogger(__name__)

class WorkflowBuilder:
    def __init__(
        self,
        genre_list: List[str],
        writer_model: ModelName = "gpt-4",
        memory_model: ModelName = "gpt-3.5-turbo",
    ):
        self.graph_builder = StateGraph(GraphState)
        self.genre_list = genre_list
        self.writer_model = writer_model
        self.memory_model = memory_model
        self._setup_agents()
        self._setup_graph()

    def _setup_agents(self):
        """Setup the agents or tools."""
        self.writer_agent = WriterAgent(
            genre_list=self.genre_list,
            model_name=self.writer_model
        )
        
        self.memory_agent = MemoryAgent(
            genre_list=self.genre_list,
            model_name=self.memory_model
        )
        
        # Create a default memory retriever - will be updated with thread ID in _retrieve_memories
        self.memory_retriever = create_memory_retriever()
        
    def _process_user_input(self, state: GraphState) -> GraphState:
        """Process user input and update the state."""
        # Add the current message to conversation history
        if state.get("current_message"):
            if "conversation_history" not in state:
                state["conversation_history"] = []
            state["conversation_history"].append(state["current_message"])
            state["current_message"] = None
        
        # Initialize error and status fields if they don't exist
        if "error" not in state:
            state["error"] = None
        if "status" not in state:
            state["status"] = "RUNNING"
            
        # Ensure genre_list is in the state
        if "genre_list" not in state and hasattr(self, "genre_list"):
            state["genre_list"] = self.genre_list
        
        # Set thread ID from configurable parameters if available
        thread_id_from_config = state.get("configurable", {}).get("thread_id")
        if thread_id_from_config:
            # Log all available parameters to help with debugging
            logger.info(f"Configurable parameters: {state.get('configurable', {})}")
            logger.info(f"Setting thread_id to: {thread_id_from_config}")
            state["thread_id"] = thread_id_from_config
            
        # If we still don't have a thread_id, use a default or warn
        if "thread_id" not in state:
            logger.warning("No thread_id found in state or configurable parameters")
            # Generate a default UUID if needed
            default_thread_id = str(uuid.uuid4())
            logger.info(f"Generated default thread_id: {default_thread_id}")
            state["thread_id"] = default_thread_id
            
        return state
    
    async def _retrieve_memories(self, state: GraphState) -> GraphState:
        """Generate memory query and retrieve relevant memories."""
        if state.get("error"):
            # Skip if we already have an error
            return state
            
        try:
            # Set thread ID in memory retriever if provided in state
            thread_id = state.get("thread_id")
            
            if not thread_id:
                logger.warning("No thread_id found for memory retrieval. Will use default collection.")
            else:
                logger.info(f"Using thread_id for memory retrieval: {thread_id}")
                # Update the retriever to use thread-specific collection
                self.memory_retriever.set_thread_id(thread_id)
                logger.info(f"Memory retriever set to thread ID: {thread_id}, using collection: {self.memory_retriever.collection_name}")
            
            # Get memory query from memory agent
            memory_query_str = await self.memory_agent.ainvoke(state)
            
            # Parse the response into a structured query
            memory_query = self.memory_retriever.parse_memory_query(memory_query_str)
            state["memory_query"] = memory_query
            
            # Initialize memories array
            memories = []
            
            # Always extract the 3 most recent messages from the conversation history
            conversation_history = state.get("conversation_history", [])
            recent_messages = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
            
            # Create memory objects from recent messages
            recent_memory_objects = []
            for i, msg in enumerate(recent_messages):
                # Determine content based on message type
                content = msg.content if hasattr(msg, "content") else str(msg)
                # Determine type (user or AI)
                msg_type = "User input" if isinstance(msg, HumanMessage) else "Story continuation"
                
                # Create a memory object from the message
                memory = {
                    "text": content,
                    "characters": [],  # We don't have character info from raw messages
                    "locations": [],   # We don't have location info from raw messages
                    "themes": [],      # We don't have theme info from raw messages
                    "importance_score": 10,  # High importance for recent context
                    "timestamp": format_rfc3339_datetime(),
                    "storyline": "recent context",
                    "emotion": "neutral",
                    "chapter": "Current Chapter",
                    "source": "recent_message",  # Tag to identify the source
                    "message_type": msg_type  # Identify if user or AI
                }
                recent_memory_objects.append(memory)
            
            try:
                # Use the memory retriever to get memories from vector database
                vector_memories = self.memory_retriever.retrieve_memories(memory_query)
                
                # Add vector memories first (they'll be displayed first)
                memories.extend(vector_memories)
                logger.info(f"Successfully retrieved {len(vector_memories)} memories from collection {self.memory_retriever.collection_name}")
                
            except Exception as e:
                # Handle the case where memory retrieval from Weaviate fails
                logger.warning(f"Error retrieving memories from collection {self.memory_retriever.collection_name}: {str(e)}")
                
                # Use conversation history as memory fallback
                # Extract up to 5 earliest messages from the conversation history
                # that are from the user (HumanMessage)
                user_messages = [msg for msg in conversation_history if isinstance(msg, HumanMessage)]
                earliest_messages = user_messages[:5]
                
                # Create memory objects from these messages
                for i, msg in enumerate(earliest_messages):
                    # Only add if not already in recent_memory_objects
                    if msg not in recent_messages:
                        # Create a memory object from the message
                        memory = {
                            "text": msg.content,
                            "characters": [],  # We don't have character info from raw messages
                            "locations": [],   # We don't have location info from raw messages
                            "themes": [],      # We don't have theme info from raw messages
                            "importance_score": 10 - i,  # Earlier messages have higher importance
                            "timestamp": format_rfc3339_datetime(),
                            "storyline": "main quest",
                            "emotion": "neutral",
                            "chapter": "Chapter 1",
                            "source": "fallback"  # Tag to identify the source
                        }
                        memories.append(memory)
                
                # Update the query to reflect what we did
                if "memory_query" in state:
                    state["memory_query"]["fallback_used"] = True
            
            # Now add the recent memories after the other memories
            # This ensures they appear last in the formatted prompt
            memories.extend(recent_memory_objects)
            
            # Add the memories to the state
            state["memory_results"] = memories
            logger.info(f"Total memories provided: {len(memories)} ({len(memories) - len(recent_memory_objects)} from DB, {len(recent_memory_objects)} from recent context)")
        
        except Exception as e:
            logger.exception(f"Error in memory query generation: {str(e)}")
            logger.warning("Continuing with empty memories")
            
            # Initialize empty results to allow the workflow to continue
            if "memory_query" not in state:
                state["memory_query"] = {"query_text": "previous conversation context", "limit": 5}
            if "memory_results" not in state:
                state["memory_results"] = []
        
        return state
    
    async def _generate_response(self, state: GraphState) -> GraphState:
        """Generate a response using the writer agent and memory context."""
        # Skip generation if we have an error, but ensure we have a valid status
        if state.get("error"):
            state["status"] = "WAITING_USER_INPUT"
            return state
            
        try:
            # Generate story continuation using the writer agent
            # Returns structured memory data (includes story text and metadata)
            memory_data = await self.writer_agent.ainvoke(state)
            
            # Extract the text content to show to the user
            story_text = memory_data.get("text", "The story continues...")
            
            # Create an AI message with just the story text
            response_message = AIMessage(content=story_text)
            
            # Add the response to the conversation history
            if "conversation_history" not in state:
                state["conversation_history"] = []
            state["conversation_history"].append(response_message)
            
            # Store the memory in the vector database
            await self._store_memory(state, memory_data)
            
            # Update state status
            state["status"] = "WAITING_USER_INPUT"
            
            logger.info("Generated story response and stored memory")
            
        except Exception as e:
            logger.exception("Error in story generation")
            state["error"] = f"Story generation error: {str(e)}"
            state["status"] = "WAITING_USER_INPUT"
        
        return state
    
    async def _store_memory(self, state: GraphState, memory_data: Dict) -> None:
        """Store a memory in the vector database.
        
        Args:
            state: Current state with thread_id
            memory_data: The memory data to store
        """
        try:
            # Verify that memory_retriever is using the correct collection
            expected_thread_id = self.memory_retriever.thread_id
            if not expected_thread_id:
                logger.warning("No thread_id set in memory_retriever during memory storage")
                # Set it from state if available
                thread_id = state.get("thread_id")
                if thread_id:
                    self.memory_retriever.set_thread_id(thread_id)
                    logger.info(f"Updated memory_retriever thread_id to {thread_id}")
            
            logger.info(f"Storing memory using thread_id: {self.memory_retriever.thread_id}, collection: {self.memory_retriever.collection_name}")
            
            # Convert decimal importance score (0-1) to integer scale (1-10) if needed
            if "importance_score" in memory_data and isinstance(memory_data["importance_score"], float) and memory_data["importance_score"] <= 1.0:
                memory_data["importance_score"] = int(memory_data["importance_score"] * 10) + 1
                # Ensure it's between 1 and 10
                memory_data["importance_score"] = min(10, max(1, memory_data["importance_score"]))
                
            # Log the memory being stored
            logger.info(f"Storing memory in collection {self.memory_retriever.collection_name}: {memory_data.get('text', '')[:50]}...")
            
            # Store the memory using the memory retriever
            memory_id = self.memory_retriever.add_memory(memory_data)
            
            if memory_id:
                logger.info(f"Successfully stored memory with ID: {memory_id} in collection {self.memory_retriever.collection_name}")
            else:
                logger.warning(f"Failed to store memory in collection {self.memory_retriever.collection_name}, no ID returned")
                
        except Exception as e:
            logger.exception(f"Error storing memory in collection {self.memory_retriever.collection_name}: {str(e)}")
            # We don't want to fail the whole workflow if memory storage fails,
            # so we just log the error and continue
    
    def _should_retrieve_memories(self, state: GraphState) -> bool:
        """Determine if we should retrieve memories."""
        # Skip memory retrieval if we already have an error
        if state.get("error"):
            return False
            
        # For now, always retrieve memories when processing input otherwise
        return True
        
    def _setup_graph(self):
        """Setup the graph flow with tool conditions."""
        # Add nodes
        self.graph_builder.add_node("process_user_input", self._process_user_input)
        self.graph_builder.add_node("retrieve_memories", self._retrieve_memories)
        self.graph_builder.add_node("generate_response", self._generate_response)
        
        # Set the entry point - fix API usage
        self.graph_builder.set_entry_point("process_user_input")
        
        # Define the flow
        self.graph_builder.add_conditional_edges(
            "process_user_input",
            self._should_retrieve_memories,
            {
                True: "retrieve_memories",
                False: "generate_response"
            }
        )
        
        self.graph_builder.add_edge("retrieve_memories", "generate_response")
        self.graph_builder.add_edge("generate_response", END)

    def compile(self):
        """Compile the workflow graph with appropriate checkpointer configuration."""
        # Create a memory saver with default configuration
        memory_saver = MemorySaver()
        
        # For configurable parameters like thread_id, we use the config parameter at runtime
        # when calling workflow.ainvoke, not here during compilation
        return self.graph_builder.compile(
            checkpointer=memory_saver
        )
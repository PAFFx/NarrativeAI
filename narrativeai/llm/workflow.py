import logging
from .states import GraphState
from .agents.writer_agent import WriterAgent
from .agents.longterm_plotter_agent import LongTermPlotterAgent
from .agents.narrative_agent import NarrativeAgent
from .agents.tools.neo4j import Neo4jTool
from .llm import ModelName
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from typing_extensions import Literal
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class WorkflowBuilder:
    def __init__(
        self,
        genre_list: List[str],
        narrative_model: ModelName = "gpt-4",
        writer_model: ModelName = "gpt-4",
        plotter_model: ModelName = "gpt-4"
    ):
        self.graph_builder = StateGraph(GraphState)
        self.genre_list = genre_list
        self.narrative_model = narrative_model
        self.writer_model = writer_model
        self.plotter_model = plotter_model
        self._setup_agents()
        self._setup_graph()

    def _setup_agents(self):
        # Initialize tools first
        #self.graph_tool = Neo4jTool()
        
        # Initialize writer agent with tools
        self.writer_agent = WriterAgent(
            genre_list=self.genre_list,
            model_name=self.writer_model
        )

        self.longterm_plotter_agent = LongTermPlotterAgent(
            genre_list=self.genre_list,
            model_name=self.plotter_model
        )

        self.narrative_agent_tools = [LongTermPlotterAgent.transfer_to_longterm_plotter]
        self.narrative_agent = NarrativeAgent(
            tools=self.narrative_agent_tools,
            genre_list=self.genre_list,
            model_name=self.narrative_model
        )

        # Add nodes
        self.graph_builder.add_node("writer", self._writer_node)
        self.graph_builder.add_node("longterm_plotter", self._longterm_plotter_node)
        self.graph_builder.add_node("narrative", self._narrative_node)

    async def _narrative_node(self, state: GraphState) -> Command[Literal["writer", "longterm_plotter", "__end__"]]:
        """Process the input through the narrative agent with tools."""
        try:
            response = await self.narrative_agent.ainvoke(state)
            
            # Check for tool calls in both response.tool_calls and additional_kwargs
            tool_calls = []

            # Handle Anthropic-style tool calls (directly in tool_calls)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls.extend(response.tool_calls)
                
            # Handle OpenAI-style tool calls (in additional_kwargs)
            if hasattr(response, 'additional_kwargs') and response.additional_kwargs.get('tool_calls'):
                tool_calls.extend(response.additional_kwargs['tool_calls'])

            if tool_calls:
                # Get the last tool call
                tool_call = tool_calls[-1]
                
                # Handle both OpenAI and Anthropic tool call formats
                if tool_call.get("name") == "transfer_to_longterm_plotter" or \
                   (tool_call.get("function", {}).get("name") == "transfer_to_longterm_plotter"):

                    # Extract act from either format
                    act = ""
                    if "args" in tool_call:  # Anthropic format
                        act = tool_call["args"].get("act", "")
                    elif "function" in tool_call:  # OpenAI format
                        import json
                        args = json.loads(tool_call["function"]["arguments"])
                        act = args.get("act", "")
                    
                    return Command(
                        goto="longterm_plotter",
                        update={"requested_act": act, "conseq_longterm_count": state.get("conseq_longterm_count", 0) + 1}
                    )

            # If no tool calls or not the expected tool, proceed to writer
            content = response.content if response.content else ""
            return Command(
                goto="writer",
                update={
                    "guidelines": [content],
                    "conseq_longterm_count": 0  # Reset counter when going to writer
                }
            )
        except Exception as e:
            logger.error(f"Error in narrative node: {str(e)}")
            return Command(goto="__end__")

    async def _writer_node(self, state: GraphState) -> GraphState:
        """Process the input through the writer agent."""
        try:
            response = await self.writer_agent.ainvoke(state)
            return {
                "stories": [AIMessage(content=response)],
            }
        except Exception as e:
            logger.error(f"Error in writer node: {str(e)}")
            return Command(goto="__end__")
    
    async def _longterm_plotter_node(self, state: GraphState) -> GraphState:
        """Process the input through the longterm plotter agent with tools."""
        try:
            runnable_config = RunnableConfig(recursion_limit=3) #Only 3 discussions are allowed per request
            response = await self.longterm_plotter_agent.ainvoke(state, runnable_config)
            return {"longterm_plots": [response]}
        except Exception as e:
            logger.error(f"Error in longterm plotter node: {str(e)}")
            return Command(goto="__end__")

    def _setup_graph(self):
        """Setup the graph flow with tool conditions."""
        # Set the entry point
        self.graph_builder.add_edge(START, "narrative")
        self.graph_builder.add_edge("longterm_plotter", "narrative")
        self.graph_builder.add_edge("writer", END)

    def compile(self):
        memory_saver = MemorySaver()
        return self.graph_builder.compile(
            checkpointer=memory_saver,
            # TODO: Add interruption points before tool usage
        )
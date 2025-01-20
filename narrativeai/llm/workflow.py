from .states import GraphState
from .agents.writer_agent import WriterAgent
from .agents.longterm_plotter_agent import LongTermPlotterAgent
from .agents.narrative_agent import NarrativeAgent
from .agents.tools.neo4j import Neo4jTool
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.runnables.config import RunnableConfig
from typing_extensions import Literal
from typing import List


class WorkflowBuilder:
    def __init__(self, genre_list: List[str]):
        self.graph_builder = StateGraph(GraphState)
        self.genre_list = genre_list
        self._setup_agents()
        self._setup_graph()

    def _setup_agents(self):
        # Initialize tools first
        #self.graph_tool = Neo4jTool()
        
        # Initialize writer agent with tools
        self.writer_agent = WriterAgent(genre_list=self.genre_list)

        self.longterm_plotter_agent = LongTermPlotterAgent(genre_list=self.genre_list)

        self.narrative_agent_tools = [LongTermPlotterAgent.transfer_to_longterm_plotter]
        self.narrative_agent = NarrativeAgent(tools=self.narrative_agent_tools, genre_list=self.genre_list)

        # Add nodes
        self.graph_builder.add_node("writer", self._writer_node)
        self.graph_builder.add_node("longterm_plotter", self._longterm_plotter_node)
        self.graph_builder.add_node("narrative", self._narrative_node)

    def _narrative_node(self, state: GraphState) -> Command[Literal["writer", "longterm_plotter", "__end__"]]:
        """Process the input through the narrative agent with tools."""
        response = self.narrative_agent.invoke(state)
        if len(response.tool_calls) > 0:
            tool_call = response.tool_calls[-1]
            if tool_call["name"] == "transfer_to_longterm_plotter":
                # Extract the act from the tool call arguments
                requested_act = tool_call["args"].get("act", "")
                return Command(
                    goto="longterm_plotter",
                    update={"requested_act": requested_act}
                )

        return Command(
            goto="writer",
            update={"guidelines": [response.content]}
        )

    def _writer_node(self, state: GraphState) -> GraphState:
        """Process the input through the writer agent."""
        response = self.writer_agent.invoke(state)
        return {"stories": [response]}
    
    def _longterm_plotter_node(self, state: GraphState) -> GraphState:
        """Process the input through the longterm plotter agent with tools."""
        runnable_config = RunnableConfig(recursion_limit=3) #Only 3 discussions are allowed per request
        response = self.longterm_plotter_agent.invoke(state, runnable_config)
        return {"longterm_plots": [response]}

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
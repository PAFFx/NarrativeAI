from .states import GraphState
from .agents.writer_agent import WriterAgent
from .agents.longterm_plotter_agent import LongTermPlotterAgent
from .models import StoryContext
from .agents.tools.neo4j import Neo4jTool
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from typing_extensions import Literal


class WorkflowBuilder:
    def __init__(self, story_context: StoryContext):
        self.story_context = story_context
        self.graph_builder = StateGraph(GraphState)
        self._setup_agents()
        self._setup_graph()

    def _setup_agents(self):
        # Initialize tools first
        #self.graph_tool = Neo4jTool()
        
        # Initialize writer agent with tools
        self.writer_agent_tools = [LongTermPlotterAgent.transfer_to_longterm_plotter]
        self.writer_agent = WriterAgent(tools=self.writer_agent_tools)

        self.longterm_plotter_agent = LongTermPlotterAgent()

        # Add nodes
        self.graph_builder.add_node("writer", self._writer_node)
        self.graph_builder.add_node("longterm_plotter", self._longterm_plotter_node)

    def _writer_node(self, state: GraphState) -> Command[Literal["longterm_plotter", "__end__"]]:
        """Process the input through the writer agent with tools."""
        response = self.writer_agent.invoke(state)
        if len(response.tool_calls) > 0:
            #TODO: only 1 tools for now

            return Command(
                goto="longterm_plotter" 
            )
        
        return {"stories": [response]}
    
    def _longterm_plotter_node(self, state: GraphState) -> GraphState:
        """Process the input through the longterm plotter agent with tools."""
        response = self.longterm_plotter_agent.invoke(state)
        return {"longterm_plots": [response]}

    def _setup_graph(self):
        """Setup the graph flow with tool conditions."""
        # Set the entry point
        self.graph_builder.add_edge(START, "writer")
        self.graph_builder.add_edge("longterm_plotter", "writer")

    def compile(self):
        memory_saver = MemorySaver()
        return self.graph_builder.compile(
            checkpointer=memory_saver,
            # Add interruption points before tool usage
        )
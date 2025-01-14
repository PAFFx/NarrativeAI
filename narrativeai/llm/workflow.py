from .states import GraphState
from .agents.writer import WriterAgent
from .models import StoryContext
from .agents.tools.neo4j import Neo4jTool
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode


class WorkflowBuilder:
    def __init__(self, story_context: StoryContext):
        self.story_context = story_context
        self.graph_builder = StateGraph(GraphState)
        self._setup_agents()
        self._setup_graph()

    def _setup_agents(self):
        # Initialize tools first
        self.graph_tool = Neo4jTool()
        self.tools = [self.graph_tool]
        
        # Initialize writer agent with tools
        self.writer_agent = WriterAgent(tools=self.tools)
        self.writer_agent.update_context(
            genre=self.story_context.genre,
            tone=self.story_context.tone
        )
        # Add nodes
        self.graph_builder.add_node("writer", self._writer_node)
        self.tool_node = ToolNode(tools=self.tools)
        self.graph_builder.add_node("tools", self.tool_node)

    def _writer_node(self, state: GraphState) -> GraphState:
        """Process the input through the writer agent with tools."""
        messages = state["messages"]
        response = self.writer_agent.invoke(messages)
        
        return {"messages": [response]}
    
    def _longterm_plotter_node(self, state: GraphState) -> GraphState:
        """Process the input through the longterm plotter agent with tools."""
        messages = state["messages"]
        response = self.longterm_plotter_agent.invoke(messages)
        return {"messages": [response]}

    def _route_tools(self, state: GraphState) -> GraphState:
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    def _setup_graph(self):
        """Setup the graph flow with tool conditions."""
        # Set the entry point
        self.graph_builder.add_edge(START, "writer")
        self.graph_builder.add_edge("writer", END)
        
        # Add conditional edges for tool usage
        self.graph_builder.add_conditional_edges(
            "writer",
            self._route_tools,
            {
                "tools": "tools",
                END: END
            }
        )
        
        # Add edge from tools back to writer
        self.graph_builder.add_edge("tools", "writer")

    def compile(self):
        memory_saver = MemorySaver()
        return self.graph_builder.compile(
            checkpointer=memory_saver,
            # Add interruption points before tool usage
            interrupt_before=["tools"]
        )
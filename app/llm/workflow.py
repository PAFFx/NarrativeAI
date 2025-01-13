from .states import GraphState
from .agents.retriever import RetrieverAgent
from langgraph.graph import START, END, StateGraph


class WorkflowBuilder:
    def __init__(self):
        self.graph_builder = StateGraph(GraphState)
        self._setup_agents()
        self._setup_graph()

    def _setup_agents(self):
        # Agent Nodes
        self.retriever_agent = RetrieverAgent()
        self.graph_builder.add_node("retriever_agent", self._retriever_agent_node)

    def _retriever_agent_node(self, state: GraphState):
        return {"messages": [self.retriever_agent.invoke(state["messages"])]}

    def _setup_graph(self):
        self.graph_builder.add_edge(START, "retriever_agent")
        self.graph_builder.add_edge("retriever_agent", END)

    def compile(self):
        return self.graph_builder.compile()
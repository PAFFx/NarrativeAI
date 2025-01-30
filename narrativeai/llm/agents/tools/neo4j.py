import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from pydantic import Field, PrivateAttr
from ...models import Neo4jQueryResult
from ...llm import get_model, ModelName

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jTool(BaseTool):
    """A tool for querying and retrieving information from Neo4j graph database."""
    
    name: str = "graph_retriever"
    description: str = """Use this tool to query the knowledge graph for information about characters, 
    locations, events, or any other story elements. The tool understands natural language questions 
    and returns relevant information from the graph database."""
    
    # Private attributes that won't be included in the schema
    _uri: str = PrivateAttr()
    _username: str = PrivateAttr()
    _password: str = PrivateAttr()
    _graph: Neo4jGraph = PrivateAttr()
    _llm: Any = PrivateAttr()
    _qa_chain: GraphCypherQAChain = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._uri = NEO4J_URI
        self._username = NEO4J_USERNAME
        self._password = NEO4J_PASSWORD
        self._graph = Neo4jGraph(
            url=self._uri,
            username=self._username,
            password=self._password
        )
        self._llm = get_model(model_name="gpt-4")
        self._qa_chain = self._setup_qa_chain()

    def _setup_qa_chain(self) -> GraphCypherQAChain:
        """Setup the GraphCypherQAChain with custom prompts."""
        # Get the schema for context
        schema = self._graph.get_schema
        
        # Create custom prompts for better Cypher generation
        cypher_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Neo4j expert. Generate Cypher queries based on user questions.
            Use only the available schema. Be precise and avoid hallucinating relationships or properties.
            
            Schema: {schema}
            
            Guidelines:
            1. Use appropriate labels and relationships from the schema
            2. Include relevant properties in the RETURN clause
            3. Use parameters for values when appropriate
            4. Keep queries efficient and focused
            
            Question: {question}
            """),
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on Neo4j query results.
            Provide clear, concise answers using only the information from the query results.
            If no results are found, clearly state that.
            
            Question: {question}
            Cypher query: {query}
            Query result: {result}
            """),
        ])

        return GraphCypherQAChain.from_llm(
            llm=self._llm,
            graph=self._graph,
            cypher_prompt=cypher_prompt,
            qa_prompt=qa_prompt,
            validate_cypher=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True  # TODO: Limit access by setting user permissions
        )

    def _run(self, query: str) -> str:
        """
        Execute a natural language query against Neo4j using GraphCypherQAChain.
        This is the main method called by the LangChain tool system.
        
        Args:
            query: Natural language question about the graph data
            
        Returns:
            String response with the answer or error message
        """
        try:
            result = self._qa_chain(query)
            return result["result"]
        except Exception as e:
            return f"Error querying the graph database: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version of _run."""
        raise NotImplementedError("Neo4jTool does not support async operations yet")

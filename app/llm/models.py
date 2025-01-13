from typing import Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class Neo4jQueryResult:
    """Model for Neo4j query results."""
    answer: str
    cypher: Optional[str] = None
    raw_results: Optional[Any] = None
    error: Optional[str] = None

    @classmethod
    def from_chain_result(cls, result: Dict) -> "Neo4jQueryResult":
        """Create from GraphCypherQAChain result."""
        return cls(
            answer=result["result"],
            cypher=result["intermediate_steps"]["query"],
            raw_results=result["intermediate_steps"]["context"]
        )

    @classmethod
    def from_error(cls, error: str) -> "Neo4jQueryResult":
        """Create an error result."""
        return cls(
            answer="I encountered an error while trying to answer your question.",
            error=str(error)
        )
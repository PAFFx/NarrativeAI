from llm.utils import get_model
from config.llm import LLMConfig

llm = get_model(LLMConfig["model"])

class RetrieverAgent:
    def __init__(self):
        self.llm = llm

    def invoke(self, query: str):
        return self.llm.invoke(query)

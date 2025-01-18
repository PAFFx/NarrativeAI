import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_model(name: str = "gpt-4o", temperature: float = 0.0):
    return ChatOpenAI(model=name, api_key=OPENAI_API_KEY, temperature=temperature)
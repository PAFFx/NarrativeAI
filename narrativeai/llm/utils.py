import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List, Union
from .states import Message
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_model(name: str = "gpt-4o", temperature: float = 0.5):
    return ChatOpenAI(model=name, api_key=OPENAI_API_KEY, temperature=temperature)

def format_conversation(messages: List[Union[BaseMessage, Message]]) -> str:
    """Format the conversation into a readable string."""
    formatted_messages = []
    for message in messages:
        if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
            role = "User" if isinstance(message, HumanMessage) else "Assistant"
            content = message.content
        elif isinstance(message, tuple):
            role, content = message
            role = role.capitalize()
        else:
            continue
        formatted_messages.append(f"{role}: {content}")

    return "\n".join(formatted_messages) if formatted_messages else "No previous conversation"

def get_message_content(message: Union[BaseMessage, Message]) -> str:
    """Extract content from different message formats."""
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        return message.content
    elif isinstance(message, tuple):
        return message[1]
    return str(message)
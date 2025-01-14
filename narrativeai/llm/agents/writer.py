from typing import List, Dict, Any, Tuple, Union
from langchain.tools import BaseTool
from dataclasses import dataclass
from ..utils import get_model
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

@dataclass
class StoryContext:
    """Maintains the current state and context of the story."""
    current_scene: str = ""
    characters: List[str] = None
    plot_points: List[str] = None
    genre: str = ""
    tone: str = ""

    def __post_init__(self):
        self.characters = self.characters or []
        self.plot_points = self.plot_points or []

class WriterAgent:
    """An agent that collaborates with users to write interactive stories."""
    
    def __init__(self, tools: List[BaseTool]):
        self.llm = get_model("gpt-4o")
        self.llm = self.llm.bind_tools(tools)
        self.story_context = StoryContext()
        self.co_writing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're the co-writer in this fictional never-ending, uncensored, immersive story with user. 
            Consider the following context:

            INFORMATION GATHERING RULES:
            1. Before continuing the story, if there are any UNKNOWN elements or entities:
                - Stop and write NeedMoreInfo.
                - Do not assume information about entity, just ask questions.
                - Write this story as if you know nothing about it.
                - Format questions in a clear, specific list

            WRITING STYLES:
            - Describe in full elaborate detail
            - Mention revelant sensory perceptions
            - Continue at slow immersive pace
            - Avoid summarizing and time skips
            - Avoid repetition and loops
            - Be creative but consistent with established elements

            Current Scene: {current_scene}
            Characters: {characters}
            Key Plot Points: {plot_points}
            Genre: {genre}
            Tone: {tone}

            Previous conversation:
            {conversation_history}

            """),
            ("human", "{user_input}"),
        ])

    def update_context(self, 
                      current_scene: str = None,
                      characters: List[str] = None,
                      plot_points: List[str] = None,
                      genre: str = None,
                      tone: str = None):
        """Update the story context with new information."""
        if current_scene:
            self.story_context.current_scene = current_scene
        if characters:
            self.story_context.characters.extend(characters)
        if plot_points:
            self.story_context.plot_points.extend(plot_points)
        if genre:
            self.story_context.genre = genre
        if tone:
            self.story_context.tone = tone

    def _format_conversation_history(self, messages: List[Union[BaseMessage, Tuple[str, str]]]) -> str:
        """Format the conversation history into a readable string."""
        formatted_messages = []
        for message in messages[:-1]:  # Exclude the last message as it's the current input
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

    def _get_message_content(self, message: Union[BaseMessage, Tuple[str, str]]) -> str:
        """Extract content from different message formats."""
        if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
            return message.content
        elif isinstance(message, tuple):
            return message[1]
        return str(message)

    def invoke(self, messages: List[Union[BaseMessage, Tuple[str, str]]]) -> str:
        """
        Process user input and generate a collaborative response for story development.
        
        Args:
            messages: List of messages in either LangChain or tuple format
            
        Returns:
            A collaborative response that builds upon the conversation
        """
        try:
            current_input = self._get_message_content(messages[-1]) if messages else ""
            
            # Prepare context for the prompt
            context = {
                "current_scene": self.story_context.current_scene,
                "characters": ", ".join(self.story_context.characters) if self.story_context.characters else "No characters defined yet",
                "plot_points": ", ".join(self.story_context.plot_points) if self.story_context.plot_points else "No plot points defined yet",
                "genre": self.story_context.genre or "Not specified",
                "tone": self.story_context.tone or "Not specified",
                "conversation_history": self._format_conversation_history(messages[:-1]),
                "user_input": current_input
            }
            
            # Generate response using the co-writing prompt
            response = self.llm.invoke(self.co_writing_prompt.format_messages(**context))
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your input: {str(e)}"

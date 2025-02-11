import os
from typing import List, Dict, Any, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage
import logging
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

logger = logging.getLogger(__name__)

ModelProvider = Literal["openai", "anthropic"]
ModelName = Literal[
    "gpt-4o",
    "claude-3-5-sonnet-20241022"
]

# Mapping from user-friendly names to actual model names
MODEL_NAME_MAPPING = {
    "gpt-4": "gpt-4o",
    "gpt-4o": "gpt-4o",  # OpenAI GPT-4
    "claude-3-sonnet": "claude-3-5-sonnet-20241022"
}

def get_model_name(user_model: str) -> ModelName:
    """Convert user-friendly model name to actual model name."""
    if user_model in MODEL_NAME_MAPPING:
        return MODEL_NAME_MAPPING[user_model]
    raise ValueError(f"Unsupported model: {user_model}. Supported models are: {', '.join(MODEL_NAME_MAPPING.keys())}")

def get_model_max_tokens(model_name: ModelName) -> int:
    """Get the maximum tokens for a model from its configuration."""
    config = LLMConfig.get_config(model_name)
    return config.get("max_tokens", 1000)  # Return max_tokens or default to 1000

class LLMConfig:
    """Configuration for LLM models."""
    
    # Default configurations for each model
    MODEL_CONFIGS = {
        "gpt-4o": {
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 16000,
            "top_p": 1.0,
        },
        "claude-3-5-sonnet-20241022": {
            "provider": "anthropic",
            "temperature": 0.7,
            "max_tokens": 4000,
            "top_p": 1.0,
        }
    }
    
    @classmethod
    def get_config(cls, model_name: ModelName) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return cls.MODEL_CONFIGS[model_name].copy()

class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(model_name: ModelName, **kwargs) -> ChatAnthropic | ChatOpenAI:
        """Create an LLM instance based on model name and configuration."""
        config = LLMConfig.get_config(model_name)
        config.update(kwargs)  # Override defaults with provided kwargs
        
        provider = config.pop("provider")
        
        if provider == "openai":
            return ChatOpenAI(
                model_name=model_name,
                api_key=OPENAI_API_KEY,
                **config
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=model_name,
                api_key=ANTHROPIC_API_KEY,
                **config
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

def get_model(
    model_name: ModelName = "gpt-4o",
    **kwargs
) -> ChatOpenAI | ChatAnthropic :
    """Convenience function to get an LLM instance."""
    return LLMFactory.create_llm(get_model_name(model_name), **kwargs)

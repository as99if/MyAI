import json
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from asyncio.log import logger

class ApiKeys(BaseSettings):
    """Settings for API keys used in the application."""
    google_maps_api_key: str = Field(..., env="GOOGLE_MAPS_API_KEY")
    google_custom_search_api_key: str = Field(..., env="GOOGLE_CUSTOM_SEARCH_API_KEY")
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    gemini_google_cloud_project: str = Field(..., env="GEMINI_GOOGLE_CLOUD_PROJECT")
    gemini_google_cloud_location: str = Field(..., env="GEMINI_GOOGLE_CLOUD_LOCATION")
    gemini_google_genai_use_vertexai: bool = Field(..., env="GEMINI_GOOGLE_GENAI_USE_VERTEXAI")
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")
    google_custom_search_engine_url: str = Field(..., env="GOOGLE_CUSTOM_SEARCH_ENGINE_URL")
    google_custom_search_engine_id: str = Field(..., env="GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
    news_api_org_api_key: str = Field(..., env="NEWS_API_ORG_API_KEY")
    
    class Config:
        """Configuration for the ApiKeys class."""
        env_file = ".env"
        env_file_encoding = "utf-8"

api_keys = ApiKeys()

def load_config(path: str = "src/config/config.json"):
    prompt_path = Path(path)
    try:
        with open(prompt_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {prompt_path}")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config file")
        raise

config = load_config()

def load_prompt(path: str = "src/config/prompts/system_prompts.json"):
    prompt_path = Path(path)
    try:
        with open(prompt_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {prompt_path}")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config file")
        raise

prompts = load_prompt()
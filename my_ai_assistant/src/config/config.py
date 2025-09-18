import json
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from asyncio.log import logger

class ApiKeys(BaseSettings):
    """Settings for API keys used in the application."""
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")
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
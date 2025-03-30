from pydantic_settings import BaseSettings, Field
from typing import Optional

class ApiKeys(BaseSettings):
    """Settings for API keys used in the application."""
    google_maps_api_key: str = Field(..., env="GOOGLE_MAPS_API_KEY")
    google_custom_api_key: str = Field(..., env="GOOGLE_CUSTOM_API_KEY")
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    tavily_search_api_key: str = Field(..., env="TAVILY_SEARCH_API_KEY")


    class Config:
        """Configuration for the settings class."""
        env_file = ".env"
        env_file_encoding = "utf-8"

api_keys = ApiKeys()
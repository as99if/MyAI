import asyncio
import pprint
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

from pydantic import BaseModel, Field
from src.inference_engine.inference_processor import InferenceProcessor
from src.ai_tools.gemini import gemini_inference
from src.ai_tools.groq import groq_inference
from src.config.config import load_config
from src.core.schemas import ToolCallResponseSchema


# Define schemas for tool arguments
class GroqInferenceArgs(BaseModel):
    message: str = Field(
        default="",
        description="Prompt to ask groq for response"
    )
    is_think_needed: bool = Field(
        default=True,
        description="Whether to return thinking steps separately"
    )

class GeminiInferenceArgs(BaseModel):
    message: str = Field(
        default="",
        description="Prompt to ask gemini for response with google search"
    )
    
class MyAIAgent:
    def __init__(self):
        pass
    
my_agent = MyAIAgent()

r = asyncio.run(my_agent.agent_inference())
pprint.pprint(r)
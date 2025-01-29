# make it a client to llama-cpp-python server

import asyncio
from datetime import datetime
import json
import multiprocessing
import pprint
from typing import Any, List, Optional
from langchain_openai import ChatOpenAI
from llama_cpp import Llama as LlamaCPP
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from pydantic import BaseModel

from src.ai_tools.groq import groq_inference
from src.ai_tools.siri_service import execute_siri_command
from src.ai_tools.gemini_google_search_maps import enhanced_query
from langchain_community.chat_models import ChatLlamaCpp
import requests
from langchain.schema import HumanMessage, SystemMessage
from src.inference_engine.inference_server import InferenceServer
from src.utils.utils import load_config

class ChatMessage(BaseModel):
    role: str
    content: str
    type: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256

class InferenceProcessor:

    def __init__(self, vsion_history_engine=None, vision_engine=None):
        
        self.config = load_config()
        self.vision_engine = vision_engine
        self.vision_history_engine = vsion_history_engine

        # Maximum number of recent messages to include in context
        self.max_context_messages = self.config.get('max_context_messages', 10)
        self.max_context_messages_day_limit = self.config.get('max_context_messages_day_limit', 5)
        self.base_url = None
        
        self.inference_client = None
        # self._start_server()
        self._initialize_llm_client()
    
    
    def _initialize_llm_client(self):
        """Initialize the appropriate LLM based on configuration"""
        if self.config.get("api") == "llama_cpp_python":
            self.inference_client = ChatOpenAI(base_url=self.base_url, model_name="llama3.2-3b-instruct", streaming=False)
            
        else:
            raise ValueError("Invalid API specified in config.")
    async def _check_inference_server_health(self):
        # call /health api
        pass
    
    async def create_chat_completion(self, messages: list = []) -> Any:
        """
        Generate chat completion with context management

        Args:
            messages: List of message dictionaries with role and content
        Returns:
            AI's response
        """
        system_messages = [
                {
                    "role": "system",
                    "content": f"{self.system_prompt['chatbot_system_prompt']} {self.system_prompt['chatbot_guidelines']}",
                    "type": "system_message",
                },
                {
                    "role": "assistant",
                    "content": "Okay..",
                    "type": "assistant_message",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        messages = system_messages.extend(messages)
        print("processed prompt unformatted:\n", json.dumps(messages, indent=2))
        
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append(SystemMessage(content=msg["content"]))
            else:
                formatted_messages.append(HumanMessage(content=msg["content"]))
        
        print("processed prompt formatted:\n", json.dumps(formatted_messages, indent=2))

        try:
            if self.config.get("api") == "llama_cpp_python":
                
                # openAI or langchain chat_completion call
                response = await self.inference_client.ainvoke(formatted_messages)
                print(response)
                

                return response



        except Exception as e:
            raise Exception(f"Error during chat completion: {e}")

    





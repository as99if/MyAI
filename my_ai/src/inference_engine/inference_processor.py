"""
Inference Processor Module

This module provides chat completion capabilities using various LLM backends.
Supports multiple inference types including standard chat, vision inference, and tool-augmented responses.
Uses async operations for better performance and supports context management.

author: {Asif Ahmed}
"""

import asyncio
from datetime import datetime
import pprint
from typing import Any, List, Optional, Dict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import aiohttp
import logging
from src.utils.log_manager import LoggingManager
from src.config.config import load_config
from src.core.api_server.data_models import MessageContent
from src.core.schemas import ToolCallResponse
import requests
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from src.utils.my_ai_utils import load_prompt
from langchain_core.tools import Tool
from langchain.agents import (
    create_openai_tools_agent,
    AgentExecutor,
)

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools


class InferenceProcessor:
    """
    Handles natural language processing using LLM backends.
    Supports chat completion, vision inference, and tool-augmented responses.
    """

    def __init__(self):
        """
        Initialize inference processor with configuration and clients.
        Loads config, sets up LLM clients, and initializes system prompts.
        """
        self.logging_manager = LoggingManager()
        self.logging_manager.add_message("Initiating - InferenceProcessor", level="INFO", source="InferenceProcessor")
        
        self.config = load_config()

        # Context management settings
        self.max_context_messages = self.config.get("max_context_messages", 10)
        self.max_context_messages_day_limit = self.config.get(
            "max_context_messages_day_limit", 5
        )

        # Agent components for tool usage
        self.agent = None
        self.agent_executor = None

        self.system_prompts = load_prompt()
        self.llm_name: str = None
        self.vlm_name: str = None
        self.llm_inference_client = None
        self.vlm_inference_client = None
        
        self._initialize_llm_client()
        self.logging_manager.add_message("Coleted Initiatiion - InferenceProcessor", level="INFO", source="InferenceProcessor")
        self.logging_manager.add_message(f"LLM: {self.config.get('llm')}\nVLM: {self.config.get('vlm')}", level="INFO", source="InferenceProcessor")
        
    def _initialize_llm_client(self) -> None:
        """
        Initialize LLM clients with optimized parameters.
        Sets up both standard language model and vision model clients.
        Uses Gemma-3 optimized parameters for inference.
        """
        try:
            # Standard LLM client initialization with optimized parameters
            self.llm_inference_client = ChatOpenAI(
                base_url=self.config.get("base_url", "http://localhost:50001/v1"),
                model_name=self.config.get("llm", "gemma-3-1b"), # or overthinker
                streaming=False,
                api_key="None",
                stop_sequences=["<end_of_turn>", "<eos>"],
                temperature=1.0,
                # repeat_penalty=1.0,
                # top_k=64,
                top_p=0.95,
                n=2,
                max_completion_tokens=512,
            )
            # TODO: in a certain time of the day or in random schedule - switch to models (overthinker and gemma-3-1b)

            # Vision model client initialization
            self.vlm_inference_client = ChatOpenAI(
                base_url=self.config.get("base_url", "http://localhost:50001/v1"),
                model_name=self.config.get("vlm", "gemma-3-4b-multimodal"),
                streaming=False,
                api_key="None",
                stop_sequences=["<end_of_turn>", "<eos>"],
                temperature=1.0,
                # repeat_penalty=1.0,
                # top_k=64,
                top_p=0.95,
                # min_p=0.01,
                n=8,
                max_completion_tokens=1024,
            )
            
            """self.agent = Agent(
                model=OpenAIChat(
                    base_url=self.config.get("base_url", "http://localhost:50001"),
                    id=self.config.get("vlm", "gemma-3-4b-multimodal"),
                ),
                tools=[DuckDuckGoTools()],
                markdown=True
            )"""
            
            
        except Exception as e:
            print("Error initializing LLM or VLM client:", e)
            logging.error(f"Error initializing LLM or VLM client: {e}")
            pass

    

    async def _check_inference_server_health(self) -> bool:
        """
        Check inference server health status.

        Returns:
            bool: True if server is healthy, False otherwise

        Raises:
            Logs errors but doesn't raise exceptions
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:50001/v1/models") as response:
                    if response.status == 200:
                        return True
                    else:
                        logging.error(
                            f"Inference server health check failed with status: {response.status}"
                        )
                        return False
        except aiohttp.ClientError as e:
            logging.error(f"Failed to connect to inference server: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during health check: {str(e)}")
            return False

    async def create_chat_completion(
        self,
        messages: List[MessageContent] = [],
        schema: Any = None,
        if_vision_inference: bool = False,
        if_camera_feed: bool = False,
        llm_name: str = None,
        vlm_name: str = None
    ) -> MessageContent:
        """
        Generate chat completion with context management.

        Args:
            messages (list): Message dictionaries with role and content
            command (str, optional): Command for tool execution
            if_tool_call (bool): Whether to use tool-augmented completion called by agent
            tool_list (list, optional): List of tools to use
            if_vision_inference (bool): Whether to use vision model

        Returns:
            MessageContent: Response message

        Raises:
            Exception: If chat completion fails
        """
        self.logging_manager.add_message("Inititating create_chat_completion", level="INFO", source="InferenceProcessor")
        self.logging_manager.add_message(f"User prompt message:\n{messages}", level="INFO", source="InferenceProcessor")
        
        if llm_name:
            # 'overthinker' = deepseek r1 distill llama 3
            self.llm_inference_client.model_name = llm_name
        if vlm_name:
            self.vlm_inference_client.model_name = vlm_name
            
        is_server_alive: bool = await self._check_inference_server_health()
        if not is_server_alive:
            return MessageContent(
                role="assistant",
                timestamp=datetime.now().isoformat(),
                content="LLM Server Not Running"
            )
        
        # TODO: in a certain time of the day or in random schedule - switch to models (overthinker and gemma-3-1b)
        
        formatted_messages: list = []

        # Format user and assistant messages (from recent and new messages )
        for msg in messages:
            if type(msg.content) is str:
                if msg.role == "user":
                    formatted_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    formatted_messages.append(AIMessage(content=msg.content))
            else:
                if type(msg.content.type) == "text":
                    if msg.role == "user":
                        formatted_messages.append(HumanMessage(content=msg.content.text))
                    elif msg.role == "assistant":
                        formatted_messages.append(AIMessage(content=msg.text))
            # handle content with image, video etc. or content with multiple content segment or multiple types
        
        self.logging_manager.add_message(f"Formatted context messages with conversation history and sstem instructional messages.", level="INFO", source="InferenceProcessor")
        # print("processed prompt formatted:\n")
        # pprint.pprint(formatted_messages)

        # print("messeages formatted")
        if schema is not None:
            print("---------------- schema provided ----------------")
            self.logging_manager.add_message(f"JSON Schema provided for formatted reply", level="INFO", source="InferenceProcessor")
            self.llm_inference_client = self.llm_inference_client.with_structured_output(schema=schema)
            ## TODO: issue with schema - not working
        try:
            print("invoking model")
            self.logging_manager.add_message(f"Invoking model", level="INFO", source="InferenceProcessor")
            # langchain OpenAI like chat_completion API response
            if if_vision_inference:  # run on with bool flag / button / checkbox in gui
                # vision inference - format message prompt, texts and other contents
                print("----------------- vlm inference -----------------")
                self.logging_manager.add_message(f"VLM inference", level="INFO", source="InferenceProcessor")
                # TODO: format vision content with prompt in my ai assistant 
                # no json schema for now
                response = await self.vlm_inference_client.ainvoke(formatted_messages)
                response = MessageContent(
                    role="assistant",
                    timestamp=datetime.now().isoformat(),
                    content=response.content,
                    metadata=response.response_metadata
                )
                self.logging_manager.add_message(f"VLM inference response: {response}", level="INFO", source="InferenceProcessor")
                return response
            
            print("----------------- basic llm inference -----------------")
            self.logging_manager.add_message(f"LLM inference", level="INFO", source="InferenceProcessor")
            response = await self.llm_inference_client.ainvoke(formatted_messages)
            # pprint.pprint(response.content)
            
            # make it MessageContent
            
            response = MessageContent(
                role="assistant",
                timestamp=datetime.now().isoformat(),
                content=response.content,
                metadata=response.response_metadata
            )
            self.logging_manager.add_message(f"LLM inference response: {response}", level="INFO", source="InferenceProcessor")
            return response

        except Exception as e:
            raise Exception(f"Error during chat completion: {e}")
            



# Test execution
if __name__ == "__main__":
    ip = InferenceProcessor()
    messages: List[MessageContent] = [
        MessageContent(
            role="user",
            content="write down one to ten in numbers",
            type="user_message",
            timestamp=datetime.now().isoformat(),
        )
    ]
    asyncio.run(ip.create_chat_completion(messages))

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
import aiohttp
import logging

import requests
from src.inference_engine.mock_llm_server import MockLLMServer
from src.utils.log_manager import LoggingManager
from src.config.config import load_config, load_prompt
from src.core.api_server.data_models import MessageContent

from src.utils.my_ai_utils import format_messages


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
        self.llm_inference_client = None
        self.server_running: bool = False

        self._initialize_llm_client()

        
        self.logging_manager.add_message("Succesfully Initiated InferenceProcessor", level="INFO", source="InferenceProcessor")

    
    def _initialize_llm_client(self) -> None:
        """
        Initialize LLM clients with optimized parameters.
        Sets up both standard language model and vision model clients.
        Uses Gemma-3 optimized parameters for inference.
        """
        try:

            # check if inference server started or not
            self.server_running = self._check_inference_server_health()
                        
            if not self.server_running:
                self.llm_inference_client = MockLLMServer()
                self.llm_server_connected = False
                self.logging_manager.add_message("Succesfully Initiated LLM Client. but, llm server connection failed", level="INFO", source="InferenceProcessor")
                self.logging_manager.add_message(f"Connected to mock LLM server", level="INFO", source="InferenceProcessor")

            else:                
                # Standard LLM client initialization with optimized parameters
                self.llm_inference_client = ChatOpenAI(
                    base_url=self.config.get("base_url", "http://localhost:50001/v1"),
                    model_name=self.config.get("llm", "overthinker"), # gemma or overthinker
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
                self.logging_manager.add_message("Succesfully Initiated LLM Client.", level="INFO", source="InferenceProcessor")
                self.logging_manager.add_message(f"LLM: {self.config.get('llm')}\nTool Call LM: {self.config.get('tool_call_lm')}", level="INFO", source="InferenceProcessor")
        
            
        except Exception as e:

            logging.error(f"Error initializing LLM client: {e}")
            self.logging_manager.add_message(f"Error initializing LLM client: {e}", level="ERROR", source="InferenceProcessor")
 

    
    def _check_inference_server_health(self) -> bool:
        """
        Check inference server health status.

        Returns:
            bool: True if server is healthy, False otherwise

        Raises:
            Logs errors but doesn't raise exceptions
        """
        try:
            response = requests.get("http://localhost:50001/v1/models", timeout=5)
            if response.status_code == 200:
                return True
            else:
                logging.error(
                    f"Inference server health check failed with status: {response.status_code}"
                )
                return False
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to connect to inference server: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during health check: {str(e)}")
            return False

    async def _check_inference_server_health_async(self) -> bool:
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
    ) -> MessageContent:
        """
        Generate chat completion with context management.

        Args:
            messages (list): Message dictionaries with role and content
            command (str, optional): Command for tool execution
            if_vision_inference (bool): Whether to use vision model

        Returns:
            MessageContent: Response message

        Raises:
            Exception: If chat completion fails
        """

        self.logging_manager.add_message("Inititating create_chat_completion", level="INFO", source="InferenceProcessor")
        # self.logging_manager.add_message(f"User prompt message:\n{messages}", level="INFO", source="InferenceProcessor")
            
        if not self.server_running:
            return MessageContent(
                role="assistant",
                timestamp=datetime.now().isoformat(),
                content="LLM Server Not Running"
            )
        formatted_messages = format_messages(messages=messages)
        
        self.logging_manager.add_message(f"Formatted context messages with conversation history and system instructional messages.", level="INFO", source="InferenceProcessor")
        # print("processed prompt formatted:\n")
        # pprint.pprint(formatted_messages)

        # print("messeages formatted")
        if schema is not None:
            self.logging_manager.add_message(f"JSON Schema provided for formatted reply", level="INFO", source="InferenceProcessor")
            self.llm_inference_client = self.llm_inference_client.with_structured_output(schema=schema)
            ## TODO: issue with schema - not working ?
        try:
            
            # langchain OpenAI like chat_completion API response
            if if_vision_inference:  # run on with bool flag / button / checkbox in gui
                # vision inference - format message prompt, texts and other contents
                self.logging_manager.add_message(f"Vision inference", level="INFO", source="InferenceProcessor")
                if if_camera_feed:
                    pass
                    # TODO: add camera feed related prompts
                else:
                    pass
                    # TODO: add other vision related prompts
                # no json schema for now
                self.logging_manager.add_message(f"Formatted vision data", level="INFO", source="InferenceProcessor")
                # formatted_messages =

            # print("invoking model")
            self.logging_manager.add_message(f"Invoking model", level="INFO", source="InferenceProcessor")
            #print("*****---FORMATTED_MESSAGES--*****")
            #pprint.pprint(formatted_messages)
            #print("*****-----*****")

            response = await self.llm_inference_client.ainvoke(formatted_messages)
            # pprint.pprint(response.content)
            
            response = MessageContent(
                role="assistant",
                type="computer_response",
                timestamp=datetime.now().isoformat(),
                content=response.content,
                metadata=response.response_metadata
            )
            self.logging_manager.add_message(f"LLM inference responded", level="INFO", source="InferenceProcessor")
            return response

        except Exception as e:
            raise Exception(f"Error during chat completion: {e}")
            
    async def close(self) -> None:
        """
        Gracefully close and clean up resources.
        
        This method should be called when shutting down the InferenceProcessor
        to ensure proper cleanup of async resources and connections.
        """
        try:
            self.logging_manager.add_message("Initiating graceful shutdown of InferenceProcessor", level="INFO", source="InferenceProcessor")
            
            # Clean up LLM client resources
            if hasattr(self.llm_inference_client, 'client') and hasattr(self.llm_inference_client.client, 'close'):
                await self.llm_inference_client.client.close()
            
            # Clean up agent resources if they exist
            if self.agent_executor is not None:
                self.agent_executor = None
            
            if self.agent is not None:
                self.agent = None
            
            # Clear client references
            self.llm_inference_client = None
            
            self.logging_manager.add_message("Successfully closed InferenceProcessor resources", level="INFO", source="InferenceProcessor")
            
        except Exception as e:
            self.logging_manager.add_message(f"Error during resource cleanup: {e}", level="ERROR", source="InferenceProcessor")
            logging.error(f"Error closing InferenceProcessor resources: {e}")

    def __del__(self):
        """
        Destructor to ensure resources are cleaned up if close() wasn't called explicitly.
        """
        if self.llm_inference_client is not None:
            try:
                # Try to run cleanup in case close() wasn't called
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If there's already a running loop, schedule the cleanup
                    asyncio.create_task(self.close())
                else:
                    # If no loop is running, run the cleanup
                    asyncio.run(self.close())
            except Exception:
                # If async cleanup fails, just clear references
                self.llm_inference_client = None
                self.agent_executor = None
                self.agent = None


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

"""
Inference Processor Module

This module provides chat completion capabilities using various LLM backends.
Supports multiple inference types including standard chat, vision inference, and tool-augmented responses.
Uses async operations for better performance and supports context management.

author: {Asif Ahmed}
"""

import asyncio
from datetime import datetime
import json
import multiprocessing
import pprint
from typing import Any, List, Optional, Dict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import aiohttp
import logging
from src.ai_tools.groq import groq_inference
from src.ai_tools.siri_service import execute_siri_command
import requests
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from src.utils.utils import load_config, load_prompt
from langchain_core.tools import Tool
from langchain.agents import (
    AgentType,
    initialize_agent,
    create_openai_tools_agent,
    AgentExecutor,
)


class ChatMessage(BaseModel):
    """
    Represents a single chat message with metadata.

    Attributes:
        role (str): Message sender role (user/assistant/system)
        content (str): The message content
        type (str): Message type identifier
    """

    role: str
    content: str
    type: str


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
        self.config = load_config()

        # Context management settings
        self.max_context_messages = self.config.get("max_context_messages", 10)
        self.max_context_messages_day_limit = self.config.get(
            "max_context_messages_day_limit", 5
        )

        # Agent components for tool usage
        self.agent = None
        self.agent_executor = None

        self._initialize_llm_client()
        self.system_prompts = load_prompt()

    def _initialize_llm_client(self) -> None:
        """
        Initialize LLM clients with optimized parameters.
        Sets up both standard language model and vision model clients.
        Uses Gemma-3 optimized parameters for inference.
        """
        # Standard LLM client initialization with optimized parameters
        self.llm_inference_client = ChatOpenAI(
            base_url=self.config.get("base_url", "http://localhost:50001"),
            model_name=self.config.get("llm", "gemma-3-1b-it-GGUF"),
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

        # Vision model client initialization
        self.vlm_inference_client = ChatOpenAI(
            base_url=self.config.get("base_url", "http://localhost:50001"),
            model_name=self.config.get("vlm", "gemma-3-4b-it-GGUF"),
            streaming=False,
            api_key="None",
            stop_sequences=["<end_of_turn>", "<eos>"],
            temperature=1.0,
            # repeat_penalty=1.0,
            # top_k=64,
            top_p=0.95,
            n=2,
            max_completion_tokens=1024,
        )

    def _initialize_agent(self) -> None:
        """
        Initialize LangChain agent with tools and system prompt.
        Sets up agent for tool-augmented responses.
        """
        agent_system_prompt = [
            SystemMessage(
                content="You are a helpful assistant that can use tools to perform calculations."
            ),
            HumanMessage(content={input}),
        ]

        tools = [
            Tool(
                func=groq_inference,
                name="groq_inference",
                description="Ask Groq for response.",
            ),
        ]

        self.agent = create_openai_tools_agent(
            self.llm_inference_client, tools, agent_system_prompt
        )
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)

    def _clean_agent(self) -> None:
        """Clean up agent resources to prevent memory leaks."""
        del self.agent
        del self.agent_executor

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
        messages: list = [],
        command: str = None,
        if_tool_call: bool = False,
        tool_list: list = None,
        if_vision_inference: bool = False,
    ) -> Any:
        """
        Generate chat completion with context management.

        Args:
            messages (list): Message dictionaries with role and content
            command (str, optional): Command for tool execution
            if_tool_call (bool): Whether to use tool-augmented completion
            tool_list (list, optional): List of tools to use
            if_vision_inference (bool): Whether to use vision model

        Returns:
            Any: Model response, format depends on completion type

        Raises:
            Exception: If chat completion fails
        """
        formatted_messages = []

        # Add system message first
        formatted_messages.append(
            SystemMessage(
                content=f"{self.system_prompts['chatbot_system_prompt']} {self.system_prompts['chatbot_guidelines']}"
            )
        )
        formatted_messages.append(AIMessage(content=f"Okay."))

        # Format user and assistant messages (from recent and new messages )
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_messages.append(AIMessage(content=msg["content"]))

        # print("processed prompt formatted:\n")
        pprint.pprint(formatted_messages)

        print("messeages formatted")
        try:

            print("invoking model")

            # langchain OpenAI like chat_completion API call

            if if_vision_inference:  # trun on with bool / button / checkbox in gui
                # vision inference
                response = await self.vlm_inference_client.ainvoke(formatted_messages)
                return response

            if if_tool_call:  # trun on with bool / button / checkbox in gui
                if command:
                    self._initialize_agent()
                    # Run the agent
                    agent_response = self.agent_executor.invoke({"input": command})

                    # show a clickable small box as reply - click here to see result (if result is big or smth) (opens a new window or smth)
                    # if result is small - show there
                    agent_response = agent_response["output"]
                    formatted_messages.append(AIMessage(content=f"{agent_response}"))
                    formatted_messages.append(
                        HumanMessage(
                            content="Now write a short reply from the agent's response."
                        )
                    )
                    self._clean_agent()

            response = await self.llm_inference_client.ainvoke(formatted_messages)
            print(response)

            return response

        except Exception as e:
            raise Exception(f"Error during chat completion: {e}")


"""
# Test execution
if __name__ == "__main__":
    ip = InferenceProcessor()
    messages = [
        {
            "role": "user",
            "content": "write down one to ten in numbers",
            "type": "user_message",
            "timestamp": datetime.now().isoformat(),
        }
    ]
    asyncio.run(ip.create_chat_completion(messages))
"""
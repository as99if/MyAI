# make it a client to llama-cpp-python server

import asyncio
from datetime import datetime
import json
import multiprocessing
import pprint
from typing import Any, List, Optional
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
from langchain.agents import AgentType, initialize_agent, create_openai_tools_agent, AgentExecutor

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

    def __init__(self):

        self.config = load_config()

        # Maximum number of recent messages to include in context
        self.max_context_messages = self.config.get('max_context_messages', 10)
        self.max_context_messages_day_limit = self.config.get(
            'max_context_messages_day_limit', 5)

        self.agent = None
        self.agent_executor = None
        # self._start_server()
        self._initialize_llm_client()
        self.system_prompts = load_prompt()
        
        

    def _initialize_llm_client(self):
        """Initialize the appropriate LLM based on configuration"""
        if self.config.get("api") == "llama_cpp_python":
            self.inference_client = ChatOpenAI(
                base_url=self.config.get('base_url', 'http://localhost:8000'),
                model_name=self.config.get('llm', 'llama3.2-3b-instruct'),
                streaming=False,
                api_key="OPENAI_API_KEY"
            )

        else:
            raise ValueError("Invalid API specified in config.")
    
    def _initialize_agent(self):
        # Create StructuredTools
        # Create a prompt template
        agent_system_prompt = [
            SystemMessage(content = "You are a helpful assistant that can use tools to perform calculations."),
            HumanMessage(content={input})
        ]

        tools = [
            Tool(
                func=groq_inference,
                name="groq_inference",
                description="Ask Groq for response."
            ),
        ]
        
        # Create an agent with the tools
        self.agent = create_openai_tools_agent(self.inference_client, tools, agent_system_prompt)
        # Create an agent executor
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)
        
    def _clean_agent(self):
        del self.agent
        del self.agent_executor
    
    async def _check_inference_server_health(self):
        """Check if inference server is healthy by calling health endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:50001/v1/models') as response:
                    if response.status == 200:
                        return True
                    else:
                        logging.error(f"Inference server health check failed with status: {response.status}")
                        return False
        except aiohttp.ClientError as e:
            logging.error(f"Failed to connect to inference server: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during health check: {str(e)}")
            return False

    async def create_chat_completion(self, messages: list = [], command: str = None, if_tool_call: bool = False, tool_list: list = None) -> Any:
        """
        Generate chat completion with context management

        Args:
            messages: List of message dictionaries with role and content
        Returns:
            AI's response
        """
        formatted_messages = []
    
        # Add system message first
        formatted_messages.append(SystemMessage(
            content=f"{self.system_prompts['chatbot_system_prompt']} {self.system_prompts['chatbot_guidelines']}"
        ))
        formatted_messages.append(AIMessage(
            content=f"Okay."
        ))
        
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
            if self.config.get("api") == "llama_cpp_python":
                print("invoking model")

                # langchain OpenAI like chat_completion API call
                
                if if_tool_call:
                    if command:
                        self._initialize_agent()                        
                        # Run the agent
                        agent_response = self.agent_executor.invoke({"input": command})
                        
                        # show a clickable small box as reply - click here to see result (if result is big or smth) (opens a new window or smth)
                        # if result is small - show there
                        agent_response = agent_response["output"]
                        formatted_messages.append(AIMessage(content=f"{agent_response}"))
                        formatted_messages.append(HumanMessage(content="Now write a short reply from the agent's response."))
                        self._clean_agent()
                        
                response = await self.inference_client.ainvoke(formatted_messages)
                print(response)

                return response

        except Exception as e:
            raise Exception(f"Error during chat completion: {e}")


# test
"""
ip = InferenceProcessor()
messages = [
    {
        "role": "user",
        "content": "write down one to ten in numbers",
        "type": "user_message",
        "timestamp": datetime.now().isoformat()
    }
]
asyncio.run(ip.create_chat_completion(messages))
"""
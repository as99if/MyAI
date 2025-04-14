import asyncio
import json
import pprint
import requests
from typing import List
from langchain_openai import ChatOpenAI
from src.utils.log_manager import LoggingManager
from src.core.api_server.data_models import MessageContent
from src.core.schemas import AgentResponse, SelfReflection, ToolCallGeneration, ToolCallResponse
from src.config.config import load_config
from src.config.config import api_keys
from src.utils.my_ai_utils import load_prompt
from datetime import datetime
# Create server parameters for stdio connection
from mcp import ClientSession, types
from mcp.client.sse import sse_client
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from groq import Groq
from src.config.config import api_keys
from langchain.tools import Tool


# migrate to MCP server and clients -_-


class MyAIAgent:
    def __init__(self):
        # Initialize components
        self.config = load_config()
        self.logging_manager = LoggingManager()
        self.tool_call_inference_client = ChatOpenAI(
            base_url=self.config.get("base_url", "http://localhost:50001/v1"),
            model_name=self.config.get("tool_call_lm", "tinyagent"),  # or overthinker
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

        self.my_ai_mcp_server_params = None
        self.tools = None
        self.system_prompts = load_prompt()
        self.agent = None

        asyncio.run(self.initialize_mcp_client())
        self.logging_manager.add_message(
            "MCP Client initiated", level="INFO", source="MyAIAgent"
        )

    
    
    async def initialize_mcp_client(self):
        self.my_ai_mcp_server_params = SSEConnection(
            url="http://localhost:50002/sse", transport="sse"
        )
        async with sse_client(url="http://localhost:50002/sse") as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # List available prompts
                # prompts = await session.list_prompts()

                # Get a prompt
                # prompt = await session.get_prompt(
                #     "example-prompt", arguments={"arg1": "value"}
                # )

                # List available resources
                # resources = await session.list_resources()

                # Get tools
                self.tools = await load_mcp_tools(session)
                # print(self.tools)
                self.available_tools_prompt = "\nThese are the available tools:\n"
                # List available tools
                for tool in self.tools:
                    self.available_tools_prompt += f"** Name: {tool.name}, Description: {tool.description}, Parameter Schema: {tool.args_schema}\n"

    def format_messages(self, messages: List[MessageContent]) -> list:
        _messages = []

        for msg in messages:
           _messages.append({"role": msg.role, "content": msg.content})
        return _messages
    
    
    async def generate_planning(self, messages: List[MessageContent]) -> MessageContent:
        self.logging_manager.add_message(
            "Invoking my_ai_inference server for task runner planning", level="INFO", source="MyAIAgent"
        )
        # Define the API endpoint
        _messages = self.format_messages(messages)

        try:
            # respond with SelfReflection json schema
            response = await self.tool_call_inference_client.ainvoke(_messages)
            self.logging_manager.add_message(
                "Agent Planning Completed", level="INFO", source="MyAIAgent"
            )
            return MessageContent(
                role="assistant",
                content=response.content
            )
            
        except requests.exceptions.RequestException as e:
            print("An error occurred in planning agent's steps:", e)
            self.logging_manager.add_message(
                "Error in inference api call", level="INFO", source="MyAIAgent"
            )
            raise 
    
    async def _groq_inference_for_tool_call(self, messages: List[MessageContent], tools: list = None, if_tool_call: bool = False):
        self.logging_manager.add_message(
            "Initiating groq inference server for generating of tool call parameters", level="INFO", source="MyAIAgent"
        )
        # Initialize Groq client and perform inference
        with Groq(api_key=api_keys.groq_api_key) as client:
            # Create chat completion request
            chat_completion = client.chat.completions.create(
                messages=messages,
                model = "deepseek-r1-distill-llama-70b",
                temperature=0.8,
                max_completion_tokens=1024,
                tools=tools,
                tool_choice="auto",
                top_p=0.95,
                stream=False,
                stop=None,
            )
        
        # Clean up client
        if client:
            del client
        self.logging_manager.add_message(
            "Success", level="INFO", source="MyAIAgent"
        )
        return chat_completion
    
    async def generate_tool_call_parameters(self, messages: list, tools: list = None, tool_choice: dict = None):
        # ISSUE: slm tool call pare na -_- eta kono kotha?
        
        self.logging_manager.add_message(
            "Initiating generation of tool call parameters", level="INFO", source="MyAIAgent"
        )
        _tools = []
        _tool_choice = None
    
        for tool in self.tools:
            _tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "properties": tool.args_schema["properties"],
                        "required": tool.args_schema["required"]
                    }
                }
            }
            _tools.append(_tool)
            if tool_choice is not None:
                if tool_choice == tool["name"]:
                    _tool_choice = tool["name"]
                
        _messages = self.format_messages(messages)
        response = await self._groq_inference_for_tool_call(messages=_messages, tools=_tools, if_tool_call=True)
        
        self.logging_manager.add_message(
            "Successfully selected tools and generated parameters for task excecution.", level="INFO", source="MyAIAgent"
        )
        return response.choices[0].message.tool_calls
    
    async def execute(
        self, user_query: str, task_context: List[MessageContent] = None
    ) -> MessageContent:

        self.logging_manager.add_message(
            "Starting tool excecution process", level="INFO", source="MyAIAgent"
        )

        # Create the message structure
        messages = [
            MessageContent(
                role="user",
                content=f"You are a helpful AI agent MyAI Agent, and you can plan to use function/tool calling to exceute a task a query."
            ),
            MessageContent(
                role="assistant",
                content=f"Okay. Tell me you query, and I will plan steps for tool callings for the task and respond with the tool calling arguments/parameters.",
            ),
            MessageContent(
                role="user",
                content=f"This is my query: {user_query} to MyAI Agent. These are the available the tools:\n{self.available_tools_prompt}.\nInstruction: " + self.system_prompts["tool_call_planning_instruction"],
            ),
        ]
        args = None
        
        plnaning_response = await self.generate_planning(messages)
        # print("\n---PLAN---\n")
        # pprint.pprint(plnaning_response)

        messages = messages + [plnaning_response]
        # print("Messages:\n")
        
        # schema = json.dumps(ToolCallGeneration.model_json_schema())
        tool_call_prompt: MessageContent = MessageContent(
                role="user",
                content=f"Now, according to the plan, select one or mutltiple tools for the query and respond with their parameters. Keep reasoning short.",
        )
        messages = messages + [tool_call_prompt]
        # pprint.pprint(_messages)
        
        generated_tool_call_parameters = await self.generate_tool_call_parameters(messages)

        
        pprint.pprint(generated_tool_call_parameters)
        tool_call_responses: List[ToolCallResponse] = []
        # generated_tool_call_parameters[0].function.arguments and .name
        for fn in generated_tool_call_parameters:
            if fn.type == "function":
                fn_name = fn.function.name
                fn_arguments = fn.function.arguments

            try:
                # Call a tool
                self.logging_manager.add_message(
                    f"Starting tool excecution process for function: {fn_name} with arguments: {fn_arguments}", level="INFO", source="MyAIAgent"
                )
                async with sse_client(url="http://localhost:50002/sse") as (read, write):
                    async with ClientSession(read, write) as session:
                        tool_result = await session.call_tool(
                            fn_name, arguments=fn_arguments
                        )

                print("-----Tool Result-----")
                pprint.pprint(tool_result)
                tool_call_response: ToolCallResponse = ToolCallResponse(
                        execution_id="ss",
                        tool_used=fn_name,
                        description="",
                        parameters=fn_arguments,
                        result=tool_result,
                    )
                tool_call_responses.append(tool_call_response)
                self.logging_manager.add_message(
                    "Successfully excecuted function: {fn_name}", level="INFO", source="MyAIAgent"
                )
            except Exception as e:
                print("-----Failed-----")
                    
                self.logging_manager.add_message(
                        "Error in tool: {e}", level="INFO", source="MyAIAgent"
                )
                tool_call_response: ToolCallResponse = ToolCallResponse(
                        execution_id="ss",
                        tool_used=fn_name,
                        description="",
                        parameters=fn_arguments,
                        result="Failed excecution.",
                    )
                tool_call_responses.append(tool_call_response)

                

        overall_tool_excecution: MessageContent = MessageContent(role="assistant", content="Tool Call Excecution: " + str(tool_call_responses))
        messages = messages + [overall_tool_excecution]

        # post self reflection for tool call result
        self.logging_manager.add_message(
            "Task executions completed", level="INFO", source="MyAIAgent"
        )
        unspoken_post_self_reflection_prompt = MessageContent(
            role="user",
            content=f"My original query was: {user_query}\n\n"
            + self.system_prompts["post_think_instruction"] + "\nAnswer to my query based on the tool call excecution.",
        )
        # add agent task memory to database from MyAIAgent.. TODO: create method and database
        # await self._add_messages_to_agent_task_hostory([agent_task_memory])

        
        messages = messages + [unspoken_post_self_reflection_prompt]
        del unspoken_post_self_reflection_prompt
        pprint.pprint(messages)

        # Generate response with post thinking on tool call result
        response = await self.tool_call_inference_client.invoke(messages)
        self.logging_manager.add_message(
            "Post self reflection from tool call completed",
            level="INFO",
            source="MyAIAgent",
        )
        messages = messages + [response]

        return response



ma = MyAIAgent()
x = asyncio.run(ma.execute(user_query="What is the current time in Tokyo? Ask Gemini"))
print(x)


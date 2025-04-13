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


# migrate to MCP server and clients -_-


class MyAIAgent:
    def __init__(self):
        # Initialize components
        self.config = load_config()
        self.logging_manager = LoggingManager()
        self.tool_call_inference_client = ChatOpenAI(
            base_url=self.config.get("base_url", "http://localhost:50001/v1"),
            model_name=self.config.get("llm", "gemma-3-1b"),  # or overthinker
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
            url="http://localhost:8000/sse", transport="sse"
        )
        async with sse_client(url="http://localhost:8000/sse") as (read, write):
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



    
    async def generate_tool_call_parameters(self, messages: list, tools: list = None, tool_choice: dict = None):
        # ISSUE: issue in inference... tool call argument generate hocche ne proper structure e.. body te jhamela - prompt eng.
        self.logging_manager.add_message(
            "Invoking inference server with api call", level="INFO", source="MyAIAgent"
        )
        # Define the API endpoint
        url = "http://localhost:50001/v1/chat/completions"  # Replace with the actual API endpoint

        _tools = []
        _tool_choice = None
    
        for tool in tools:
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
                
        pprint.pprint(_tools)
        _messages = []
        for msg in messages:
           _messages.append({"role": msg.role, "content": msg.content})
        
        # Define the request body
        
        body = {
            "model": "gemm-3-1b",
            "messages": _messages,
            "tools": _tools,
            
        }
        """
        "tool_choice": {
                "type": "function",
                "function": {
                    "name": _tool_choice
                }
            }
        """

        # Make the API call
        try:
            # response = requests.post(url, json=body)
            self.tool_call_inference_client = self.tool_call_inference_client.with_structured_output(schema=ToolCallGeneration.model_json_schema)
            
            
            response = await self.tool_call_inference_client.invoke(_messages)
            # Print the response status and content
            print("--- Response ---")
            # print("Response JSON:", response.json())
            print(response)
            self.logging_manager.add_message(
                "Got response", level="INFO", source="MyAIAgent"
            )
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print("An error occurred:", e)
            self.logging_manager.add_message(
                "Error in inference api call", level="INFO", source="MyAIAgent"
            )
            raise 
    
    async def execute(
        self, user_query: str, self_reflection: SelfReflection = None
    ) -> MessageContent:

        self.logging_manager.add_message(
            "Starting tool excecutions", level="INFO", source="MyAIAgent"
        )

        # response: AgentResponse = AgentResponse(task_id="#", excecution_list=[])  # TODO: short uid
        tool_call_responses: List[ToolCallResponse] = []

        # Create the message structure
        messages = [
            MessageContent(
                role="user",
                content=f"You are a helpful AI assistant who can use function/tool calling to exceute a task. You have these function tools to use:\n{self.tools}"
            ),
            MessageContent(
                role="assistant",
                content="Okay. Tell me you query, and about the tool.",
            ),
            MessageContent(
                role="user",
                
                content=f"This was my original query: {user_query}.\n\nThis is your pre planning thoughts for responding to the task:\n",
            ),  # + {self_reflection}),
            MessageContent(
                role="assistant",
                content="Got it.",
            ),
        ]
        args = None
        if self_reflection is not None:
            # check if multiple tool call necessary (from self-replection)
            """if self_reflection.is_multiple_tool_call_necessary:
                # check suggested tools and execute each one
                for index, tool in iter(self_reflection.suggested_tools):

                    # generate arguments for that tool
                    tool_call_prompt: MessageContent = MessageContent(
                        role="user",
                        content=f"Use {tool} for the task at hand based on my query and your planning.",
                        timestamp=datetime.now().isoformat(),
                        unspoken_message=True,
                    )
                    messages = messages + [tool_call_prompt]

                    _tool = None  # get the tool (Tool) object from tools based on tool.name

                    self.tool_call_inference_client.bind_tools(tool)
                    generated_tool_call_parameters = await self.tool_call_inference_client.invoke(
                        messages=messages,
                        # tools=self.tools,
                        # tool_choice=_tool  # Specify the tool to use for extracting data
                    )
                    args = generated_tool_call_parameters.content  # or smthing

                    try:
                        # Call a tool
                        async with sse_client(url="http://localhost:8000/sse") as (read, write):
                            async with ClientSession(read, write) as session:
                                tool_result = await session.call_tool(
                                    "tool-name", arguments=args
                                )
                        # Create and run the agent
                        # self.agent = create_react_agent(
                        #     model=self.tool_call_inference_client,
                        #     tool=_tool,
                        #     response_format=ToolCallResponse
                        # )
                        # tool_result: ToolCallResponse = await self.agent.ainvoke({"messages": messages})
                        self.logging_manager.add_message(
                            "Successfully excecuted: {tool.name}",
                            level="INFO",
                            source="MyAIAgent",
                        )

                    except Exception as e:
                        tool_result: ToolCallResponse = ToolCallResponse(
                            tool_used=tool.name,
                            description=tool.description,
                            parameters=args,
                            result="Failed excecution.",
                        )
                        self.logging_manager.add_message(
                            "Error in tool: {e}", level="INFO", source="MyAIAgent"
                        )

                    tool_response: AIMessage = AIMessage(
                        content=tool_result
                    )
                    messages = messages + [tool_response]
                    tool_call_responses.append(tool_result)"""

        else:
            print("Messages:\n")
            tool_call_prompt: MessageContent = MessageContent(
                role="user",
                content="Select a tool, and use that tool for the task at hand based on my query and your planning.",
            )
            _messages = messages + [tool_call_prompt]
            pprint.pprint(_messages)
            
            generated_tool_call_parameters = await self.generate_tool_call_parameters(_messages, tools=self.tools)
            
            print(generated_tool_call_parameters)

            args = generated_tool_call_parameters["choices"][0]  # or smthing
            print(args)
            

            return "sss"
            try:
                # Call a tool
                async with sse_client(url="http://localhost:8000/sse") as (read, write):
                    async with ClientSession(read, write) as session:
                        tool_result = await session.call_tool(
                            "tool-name", arguments=args
                        )

                print("-----Tool Result-----")
                pprint.pprint(tool_result)

                self.logging_manager.add_message(
                        "Successfully excecuted: tool", level="INFO", source="MyAIAgent"
                )
            except Exception as e:
                print("-----Failed-----")
                
                self.logging_manager.add_message(
                    "Error in tool: {e}", level="INFO", source="MyAIAgent"
                )
                tool_result: ToolCallResponse = ToolCallResponse(
                    execution_id="ss",
                    tool_used="",
                    description="",
                    parameters=args,
                    result="Failed excecution.",
                )

            tool_response: MessageContent = MessageContent(role="assistant", content=str(tool_result))
            messages = messages + [tool_response]
            tool_call_responses.append(tool_result)

        # post self reflection for tool call result
        self.logging_manager.add_message(
            "Task executions completed", level="INFO", source="MyAIAgent"
        )
        unspoken_post_self_reflection_prompt = MessageContent(
            role="user",
            content=f"My original query was: {user_query}\n\n"
            + self.system_prompts["post_think_instruction"],
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

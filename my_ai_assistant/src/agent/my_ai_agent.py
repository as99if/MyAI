import asyncio
import json
import pprint
import time
from groq import Groq
from mistralai import Mistral
import requests
from typing import Any, List
from langchain_openai import ChatOpenAI
from src.agent.schemas import ThinkingSchema, ToolCallPlan, ToolCallGeneration, ToolCallResponse, ToolCallStep
from src.utils.log_manager import AgentLoggingManager
from src.core.api_server.data_models import MessageContent
from src.config.config import load_config, load_prompt
from src.config.config import api_keys
from src.utils.my_ai_utils import format_messages
from datetime import datetime
# Create server parameters for stdio connection
from mcp import ClientSession, types
from mcp.client.sse import sse_client
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from src.config.config import api_keys
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

class MyAIAgent:
    def __init__(self):
        # Initialize components
        self.config = load_config()
        self.logging_manager = AgentLoggingManager()
        self.local_agent_inference_client = ChatOpenAI(
            base_url=self.config.get("base_url", "http://localhost:50001/v1"),
            model_name=self.config.get("tool_call_lm", "overthinker"),
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

        self.my_ai_mcp_server_params = None
        self.tools = None
        self.system_prompts = load_prompt()
        self.agent = None
        self.connected: bool = False


        # asyncio.create_task(self.initialize_mcp_client())
        # self.logging_manager.add_message(
        #     "MCP Client initiated", level="INFO", source="MyAIAgent"
        # )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.initialize_mcp_client())
            self.logging_manager.add_message(
                "MCP Client initialization scheduled", level="INFO", source="MyAIAgent"
            )
        except RuntimeError:
            # No running loop yet (e.g., constructed in sync context like Qt startup)
            self.logging_manager.add_message(
                "No running event loop; defer MCP init. Later call: await my_ai_agent.initialize_mcp_client()",
                level="WARNING",
                source="MyAIAgent",
            )

    
    
    async def initialize_mcp_client(self):
        try:
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
                    self.connected = True
                    self.logging_manager.add_message(
                        "MCP Client connected and tools loaded", level="INFO", source="MyAIAgent"
                    )
        except Exception as e:
            self.connected = False
            self.logging_manager.add_message(
                f"Error initializing MCP client: {e}", level="ERROR", source="MyAIAgent"
            )


    async def local_agent_inference(self, messages: List[MessageContent], if_tool_call: bool = False, json_mode: bool = False) -> MessageContent:
        try:
            _messages = format_messages(messages)

            if if_tool_call:
                local_agent_inference_client_for_tool_call = self.local_agent_inference_client.bind_tools(self.tools)
                local_agent_inference_client_for_tool_call = self.local_agent_inference_client.with_structured_output(method="json_mode")
                

            if json_mode:
                local_agent_inference_client_for_thinking_and_planning = self.local_agent_inference_client.with_structured_output(method="json_mode")
                response = await local_agent_inference_client_for_thinking_and_planning.ainvoke(_messages)
            else:
                response = self.local_agent_inference_client.ainvoke(_messages)
            return MessageContent(
                role="assistant",
                content=str(response),
                type="local_agent_response",
                timestamp=datetime.now().isoformat()
            )
        except requests.exceptions.RequestException as e:
            # print("An error occurred in planning agent's steps:", e)
            self.logging_manager.add_message(
                "Error in inference api call", level="INFO", source="MyAIAgent"
            )
            raise

    async def agent_inference(self, messages: List[MessageContent], service_provider: str = "mistral", if_tool_call: bool = True) -> ToolCallGeneration:
        start_time = time.time()
        
        model = ""
        tool_calls = []
        client = None
        if service_provider == "groq":
            # Initialize Groq client and perform inference
            with Groq(api_key=api_keys.groq_api_key) as client:
                # Get available models and filter for Meta models with large context windows
                models = client.models.list()
                models = [model for model in models.data if "Meta" in model.owned_by and int(model.context_window) >= 32768]
                models = sorted(models, key=lambda x: x.created, reverse=True)   
                self.logging_manager.add_message(f'Groq: Using model - {models[0].id}', level='INFO', source="MyAIAgent")
                # Create chat completion request
                chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0.8,
                        max_completion_tokens=1024,
                        tools=self.tools,
                        tool_choice="auto",
                        top_p=0.95,
                        stream=False,
                        stop=None,
                        parallel_tool_calls=True
                )
            if if_tool_call:
                tool_calls = chat_completion.choices[0].message.tool_calls
            
            response = MessageContent(
                role="assistant",
                content=str(chat_completion.choices[0].message),
                type=f"online_agent_{service_provider}_response",
                timestamp=datetime.now().isoformat()
            )
            # print(response)
            
        elif service_provider == "mistral":
            with Mistral(
                api_key=api_keys.mistral_api_key,
            ) as client:
                model = "mistral-small-latest"
                self.logging_manager.add_message(f'Mistral-AI: Using model - {model}', level='INFO', source="MyAIAgent")

                chat_completion = client.chat.complete(
                        model=model,
                        messages=messages,
                        tools=self.tools,
                        tool_choice="any",
                        parallel_tool_calls=True,
                )
            if if_tool_call:
                tool_calls = chat_completion.choices[0].message.tool_calls
            response = response = MessageContent(
                role="assistant",
                content=str(chat_completion.choices[0].message),
                type=f"online_agent_{service_provider}_response",
                timestamp=datetime.now().isoformat()
            )

        elif service_provider == "my_ai":
            # not recommended
            self.logging_manager.add_message(
                f"Invoking MyAIAggent inference client to generate tool call parameters.", level="INFO", source="MyAIAgent"
            )
            # use self.local_agent_inference
                ### ISSUE: langchain ChatOpenAI client is unable do generate tool call from llama-cpp-python server
            # TODO: test bigger model
            
            try:
                # TODO: format inference client or prompt with tool call response schema
                response = self.local_agent_inference(messages, if_tool_call=if_tool_call)
                if if_tool_call:
                   tool_calls = response.content.tool_calls
                   tool_calls = []
                response = response.content
            except Exception as e:
                self.logging_manager.add_message(
                    f"Error generating tool call parameters by MyAIAgent inference client: {e}", level="INFO", source="MyAIAgent"
                )

        end_time = time.time()
        propagation_delay = end_time - start_time
        self.logging_manager.add_message(
            f"Successfully generated tool call parameters, by {service_provider}.\nPropagation delay: {propagation_delay:.2f} seconds", level="INFO", source="MyAIAgent"
        )
        
        del client
        if if_tool_call:
            return ToolCallGeneration(
                    tool_calls=tool_calls,
                    metadata={
                        "service_provider": service_provider,
                        "model": model
                    }
                )
        else:
            return response
    
    async def generate_planning(self, user_query: str, task_context: List[MessageContent] = None) -> MessageContent:
        
        messages = []
        # putting conversation history for thinking before planning
        self.logging_manager.add_message("Thinking before planning", level="INFO", source="MyAIAssistant")

        # Instantiate the output parser for thinking
        output_parser = PydanticOutputParser(pydantic_object=ThinkingSchema)
        _message = MessageContent(
            role="user",
            timestamp=datetime.now().isoformat(),
            content= "User message: " + user_query + "\nInstruction:\n"
                + self.system_prompts["think_instruction_for_action_plan"] 
                + "\n"
                + output_parser.get_format_instructions(),
            type="reminder_to_self_reflect",
            unspoken_message=True
        )
        if task_context is not None:
            messages = task_context + [_message]

        response = self.local_agent_inference(messages, json_mode=True)
        
        self_reflection_validated = ThinkingSchema.model_validate_json(response.content)
        if not self_reflection_validated:
            self.logging_manager.add_message("Error in validating thoughts with schema", level="ERROR", source="MyAIAgent")
            return MessageContent(
                role="assistant",
                timestamp=datetime.now().isoformat(),
                content= "Failed to think for task excecution",
                type="failure_message_by_agent",
                unspoken_message=True
            )
        
        print("\n\n***Agent thougthts before planning***")
        pprint.pprint(response)
        print("\n\n")

        refined_query: str = response.content.refined_query

        self.logging_manager.add_message(
            "Agent thinking Completed.", level="INFO", source="MyAIAgent"
        )
        self.logging_manager.add_message(
            "Invoking MyAIAgent task planner", level="INFO", source="MyAIAgent"
        )
        
        # not putting conversation history for planning, only keeping the 'thoughts' 
        messages = [
            MessageContent(
                role="user",
                content=f"You are a helpful AI agent MyAI Agent, and you can plan to use function/tool calling to exceute a task a query. You do not excecute any funciont/tool call, do calculation or write code.\nUser Query: {user_query}"
            ),
            MessageContent(
                role="assistant",
                content=f"Okay. These are my thoughts or reasoning on it:\n {json.dumps(response)}",
            )
        ]

        # Instantiate the output parser for planning
        output_parser = PydanticOutputParser(pydantic_object=ToolCallPlan)
        messages.append(
            MessageContent(
                role="user",
                content=f"These are the available the tools:\n{self.available_tools_prompt}.\n" +
                        f"Instruction: " + self.system_prompts["tool_call_planning_instruction"] + "\n" +
                        output_parser.get_format_instructions(),
            )
        )

        response = self.local_agent_inference(messages, json_mode=True)
        plan_validated = ToolCallPlan.model_validate_json(response.content)
        if not plan_validated:
            self.logging_manager.add_message("Error in planning for tasks with schema", level="ERROR", source="MyAIAgent")
            return MessageContent(
                role="assistant",
                timestamp=datetime.now().isoformat(),
                content= "Failed to plan for task excecution",
                type="failure_message_by_agent",
                unspoken_message=True
            )
        
        print("\n\n***Agent plans after planning***")
        pprint.pprint(response)
        print("\n\n")

        self.logging_manager.add_message(
            "Agent Planning Completed.", level="INFO", source="MyAIAgent"
        )
        self.logging_manager.add_message(
            f"Query: {refined_query}\nPlan:\n{response}", level="INFO", source="MyAIAgent"
        )
        
        return response, refined_query
        
        
    async def generate_tool_call_parameters(self, user_query: str, task_execution_plan: MessageContent) -> ToolCallGeneration:

        self.logging_manager.add_message(
            "Initiating generation of tool call parameters", level="INFO", source="MyAIAgent"
        )
        messages = [
            MessageContent(
                role="user",
                content=f"You are a helpful AI agent MyAI Agent, and you can use function/tool calling (generate parameters) to exceute a task based on a query with a plan."
                + "\nUser Query: {user_query}"
            ),
            task_execution_plan,
            MessageContent(
                role="user",
                content=f"Now, excecute tool call or generate parameters for the tool call excecutions."
            )
        ]

        _tools = []
    
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

                
        _messages = format_messages(messages)
        response = await self.agent_inference(messages=_messages, tools=_tools, service_provider='mistral')

        
        self.logging_manager.add_message(
            "Successfully selected tools and generated parameters for task excecution.", level="INFO", source="MyAIAgent"
        )
        return response
    
    async def execute(
        self, user_query: MessageContent, task_context: List[MessageContent] = None
    ) -> MessageContent:
        """
        Put 'user_query' or 'task_context'.
        Make sure 'task_context' : List[MessageContent] starts with 'user' message, and ends with 'user' message, ends with 'assitant' message.
        And 'task_context' does not contain the 'user_query'. The 'task_context' will only stay in the planning stage. 
        It will not go to tool call inference and task excecutions. 

        Args:
            user_query (str): User's query about task.
            task_context (List[MessageContent], optional): Recent conversation history before the user query. Defaults to None.

        Returns:
            MessageContent: Response by MyAI after MyAIAgent finishes excecuting task/tasks.
        """
        self.logging_manager.add_message(
            "Starting tool excecution process", level="INFO", source="MyAIAgent"
        )
        
        excecution_history: List[Any] = []

        _user_query = None
        if type(user_query.content) == str:
            _user_query = user_query.content
        elif type(user_query.content) == dict:
            if user_query.content.type == "text":
                _user_query = user_query.content.text


        task_execution_plan: ToolCallPlan = None
        # Think and Plan first
        task_execution_plan, refined_query = await self.generate_planning(user_query=_user_query, task_context=task_context)
        excecution_history.append(task_execution_plan)

        # Generate tool calls and parameters
        generated_tool_call_parameters: ToolCallGeneration = await self.generate_tool_call_parameters(user_query=refined_query, task_execution_plan=task_execution_plan)
        excecution_history.append(generated_tool_call_parameters)
        
        self.logging_manager.add_message(
            f"Refined query: {refined_query},\nGenerated Tool Calls:\n{generated_tool_call_parameters.tool_calls}", level="INFO", source="MyAIAgent"
        )

        # pprint.pprint(generated_tool_call_parameters)
        tool_call_responses: List[ToolCallResponse] = []
        # generated_tool_call_parameters[0].function.arguments and .name
        # step by step
        for fn in generated_tool_call_parameters.tool_calls:
            if fn.function:
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
                excecution_history.append(tool_call_response)
                self.logging_manager.add_message(
                    f"Successfully excecuted function: {fn_name}", level="INFO", source="MyAIAgent"
                )
            except Exception as e:
                # print("-----Failed-----")
                    
                self.logging_manager.add_message(
                        f"Error in tool: {e}", level="INFO", source="MyAIAgent"
                )
                tool_call_response: ToolCallResponse = ToolCallResponse(
                        execution_id="ss",
                        tool_used=fn_name,
                        description="",
                        parameters=fn_arguments,
                        result="Failed excecution.",
                    )
                tool_call_responses.append(tool_call_response)
                excecution_history.append(tool_call_response)

        overall_tool_excecution = MessageContent(role="assistant", content="Task: " + json.dumps(tool_call_responses))
        task_messages = task_messages + [overall_tool_excecution]

        # post self reflection for tool call result
        self.logging_manager.add_message(
            "Task executions completed", level="INFO", source="MyAIAgent"
        )


        conclusive_messages = [
            MessageContent(
                role="user",
                type="system_prompt",
                content="You are a helpful AI Agent. You can summarize, and create conclusion to one tool call or multi step tool calls. Write in bullte points, be concise and focused. Do no hallucinate."
            ),
            MessageContent(
                role="assistant",
                type="agent_reply",
                content="Okay.",
                unspoken_message=True
            ),
            MessageContent(
                role="user",
                content=f"My original query was: {user_query}\n\n"
                + f"Refined Query: {refined_query}"
                + f"Task planning: {task_execution_plan}",
                type="user_message",
                unspoken_message=True
            ),
            MessageContent(
                role="assistant",
                type="agent_reply",
                content=f"I have ther results for all the tasks or tool calls according to plan. After performing, the overall task excecution list:" 
                + "\n{overall_tool_excecution}",
                unspoken_message=True
            ),
            MessageContent(
                role="user",
                type="user_message",
                content="Write a conclusive reply to my query based on the task exceuctions. Writing in bullet points, being focused on the topic, keeping it short and relatable choices of words in response is encoraged.",
                unspoken_message=True
            )
        ]
        
        _messages = format_messages(conclusive_messages)
        response = await self.agent_inference(messages=_messages, service_provider='mistral', if_tool_call=False)
        response = MessageContent(
            role="assistant",
            content=response.content,
            type="agent_response"
        )
        self.logging_manager.add_message(
            "Post self reflection from tool call completed",
            level="INFO",
            source="MyAIAgent",
        )
        task_messages = task_messages + [response]

        return response


if __name__ == "__main__":

    ma = MyAIAgent()
    x = asyncio.run(ma.execute(user_query="What is the current time in Tokyo? Ask Gemini"))
    print(x)


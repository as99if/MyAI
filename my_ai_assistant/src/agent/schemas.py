
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages.tool import ToolCall




class ToolCallStep(BaseModel):
    step_number: int = Field(description="Step index")
    tool_name: str = Field(description="Name of the tool to use")
    tool_query: str = Field(description="A query for this tool")
    purpose: str = Field(description="Purpose of using this tool")
    expected_output: str = Field(description="Output is expected from this tool")

class ToolCallPlan(BaseModel):
    task_description: str = Field(description="Task description based on user's query.")
    planned_steps: List[ToolCallStep] = Field(description="Step by step plans to complete the task with tools")
    reasoning: str = Field(description="Explanation of why this plan is appropriate")


class ThinkingSchema(BaseModel):
    user_query: str = Field(
        description="User query for agent to act on. It is the exact user message.",
        min_length=10,
        max_length=2000,
        example="Based on the user's request, I need to analyze system performance."
    )
    refined_query: str = Field(
        description="Refine the user message to make it focused and meaningful.",
        min_length=10,
        max_length=2000,
        example="Based on the user's request, I need to analyze system performance."
    )
    thought: str = Field(
        description="AI's plan, reasoning, thoughts or self reflection as a response on the current situation, query and task",
        min_length=10,
        max_length=2000,
        example="Based on the user's request, I need to analyze system performance and search ways to optimize performance online. So, I need usage of a tool which can monitor system performace and give me proper data. For this, I will use local agent. Then I need search results from the internet for optiisation. For this, I will use online agent."
    )


class ToolCallGeneration(BaseModel):
    tool_calls: List[Any] = Field(description="A list of tool calls.")
    metadata: Optional[dict] = None

class ToolCallResponse(BaseModel):
    execution_id: str
    tool_used: str = Field(
        description="Name of the tool that was used",
        example="system_monitor"
    )
    parameters: Any = Field(
        description="Generated parameters ot the tool call",
    )
    description: str = Field(
        description="Brief description of the tool's purpose",
        example="Monitors system resources and performance metrics"
    )
    result: str = Field(
        description="Output or result from the tool execution",
        example="CPU usage: 45%, Memory: 6.2GB used"
    )
    
class AgentResponse(BaseModel):
    task_id: str = Field(
        description="An uniqu id for the excecution of task with one or multiple tool calls. The tool call history is saved with this excecution_id as key",
        default="#"
    )
    excecution_list: List[ToolCallResponse] = Field(
        description="List of tools called for the task with their results",
    )

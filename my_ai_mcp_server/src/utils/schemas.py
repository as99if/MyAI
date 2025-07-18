from pydantic import BaseModel, Field
from typing import Any, List, Optional

class SelfReflection(BaseModel):
    self_reflection: str = Field(
        description="AI's response on self reflection or response planning on the current situation and reasoning",
        min_length=10,
        max_length=2000,
        example="Based on the user's request, I need to analyze system performance"
    )
    is_tool_call_necessary: bool = Field(
        description="Indicates whether a tool call to agent is required to complete a task came up in self reflection or planning.",
        example=True
    )
    is_multiple_tool_call_necessary: bool = Field(
        description="Indicates whether the idea of multiple consicutive tool calls by agent is required to complete the tasks came up in self reflection or planning.",
        example=True
    )
    suggested_tools: List[str] = Field(
        description="List of tools from available tools that could be useful for the current task by agent. If multiple tool call is necessary (is_multiple_tool_call_necessary=True) then, it is all the necessary tools.",
        min_items=1,
        example=["system_monitor", "file_analyzer"]
    )

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
        min_length=10,
        max_length=200,
        example="Monitors system resources and performance metrics"
    )
    result: str = Field(
        description="Output or result from the tool execution",
        min_length=1,
        example="CPU usage: 45%, Memory: 6.2GB used"
    )
    
class AgentResponse(BaseModel):
    task_id: str = Field(
        description="An uniqu id for the excecution of task with one or multiple tool calls. The tool call history is saved with this excecution_id as key",
    )
    excecution_list: List[ToolCallResponse] = Field(
        description="List of tools called for the task with their results",
    )

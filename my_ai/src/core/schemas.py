from pydantic import BaseModel, Field
from typing import List, Optional

class SelfReflectionSchema(BaseModel):
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
        description="Indicates whether a multiple consicutive tool calls to agent is required to complete the tasks came up in self reflection or planning.",
        example=True
    )
    suggested_tools: List[str] = Field(
        description="List of tools from available tools that could be useful for the current task by agent. If multiple tool call is necessary (is_multiple_tool_call_necessary=True) then, it is all the necessary tools.",
        min_items=1,
        example=["system_monitor", "file_analyzer"]
    )
    preferable_commands_for_tool_call: Optional[List[str]] = Field(
        default=None,
        description="List of probable and specific commands to execute tools on the context. Do not hallucinate.",
        example=["check_cpu_usage --interval=5", "check_cpu_usage --interval=10","Search on google - What is the current situation on Quantum Computing advancement."]
    )

class ToolCallResponseSchema(BaseModel):
    tool_used: str = Field(
        description="Name of the tool that was used",
        min_length=1,
        max_length=100,
        example="system_monitor"
    )
    short_description_of_the_tool: str = Field(
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
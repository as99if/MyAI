from pydantic import BaseModel
from typing import List, Optional

class SelfReflectionSchema(BaseModel):
    self_reflection: str
    is_tool_call_necessary: bool
    preferabe_tools: List[str]
    command_for_tool_call: Optional[str] = None

class ToolCallResponseSchema(BaseModel):
    tool_used: str
    short_description_of_the_tool: str
    result: str
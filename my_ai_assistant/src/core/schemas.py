from pydantic import BaseModel, Field
from typing import Any, List, Optional

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
    tasks: list


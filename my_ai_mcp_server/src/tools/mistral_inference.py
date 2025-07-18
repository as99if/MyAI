import json
import pprint
from typing import Any
from src.utils.schemas import ToolCallResponse
from src.config.config import api_keys
from mistralai import Mistral
from src.utils.log_manager import LoggingManager
from datetime import datetime



async def mistral_inference(message: str, system_message: str = None, task_memory_messages: list[Any] = []) -> ToolCallResponse:
    logging_manager = LoggingManager()
    
    # Initialize the messages list with system prompt and initial assistant acknowledgment
    if system_message is None:
        system_message = "You are helpful AI chatbot. Respond to messages, thouroughly. Elaborate if necesssary. Bullet points, or markdown formatted text reponse is encouraged."
    messages = [
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": "Okay."},
            ]
    # Process conversation history if provided
    if len(task_memory_messages) > 0:  
        for mem_seg in task_memory_messages:
            if mem_seg.role == "user":
                messages.append({"role": "user", "content": mem_seg.content})
            elif mem_seg.role != "assistant":
                # Format non-assistant messages as structured memory segments
                temp = {
                    "role": mem_seg.role,
                    "type": mem_seg.type, 
                    "content": mem_seg.content,
                    "timestamp": mem_seg.timestamp
                }
                messages.append({"role": "assistant", "content": f"Memory segments - {mem_seg.timestamp}:\n {json.dumps(temp, indent=2)}"})
            elif mem_seg.role == "assistant":
                messages.append({"role": "assistant", "content": mem_seg.content})
    # Add the current user message
    messages.append({"role": "user", "content": message})
    with Mistral(
            api_key=api_keys.mistral_api_key,
        ) as client:
            model = "mistral-small-latest"
            chat_completion = client.chat.complete(
                    model=model,
                    messages=messages,
                    tool_choice="any",
                    parallel_tool_calls=True,
            )
    response = chat_completion.choices[0].message.content

    
    logging_manager.add_message(
        f"Successfully generated tool call parameters, by mistral.", level="INFO", source="MyAIAgent"
    )
    del client
    return ToolCallResponse(
            tool_calls=response,
            metadata={
                "service_provider": "mistral",
                "model": model
            }
        )
        
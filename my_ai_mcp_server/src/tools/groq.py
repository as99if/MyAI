"""
This module provides a method for performing inference using the Groq API with support for conversation memory and thinking steps.
It selects the most recent llm by (or create using any llm by) Meta with larger context window.
author: {Asif Ahmed}
"""

from asyncio.log import logger
import json
import os
from pathlib import Path
import pprint
from typing import Any
from groq import Groq
from src.utils.schemas import ToolCallResponseSchema
from src.config.config import api_keys
from langchain.tools import Tool



def groq_inference(
    message: str, 
    temperature: int = 0.8, 
    max_completion_tokens: int = 4096, 
    is_think_needed: bool = True,
    task_memory_messages: list[Any] = []
) -> ToolCallResponseSchema:
    """
    Performs inference using the Groq API with support for conversation memory and thinking steps.
    
    Args:
        message (str): The user's input message to process
        system_message (str): The system prompt that defines the AI's behavior
        api_key (str): Groq API authentication key
        temperature (int, optional): Controls randomness in the response. Defaults to 0.8
        max_completion_tokens (int, optional): Maximum tokens in the response. Defaults to 4096
        is_think_needed (bool, optional): Whether to return thinking steps separately. Defaults to True
        task_memory_messages (list, optional): Previous conversation history or something else as task context. Defaults to empty list. Caution - Personal data will go to Groq.
    
    Returns:
        tuple[str, str]: A tuple containing:
            - Either a string response or dict with 'think' and 'response' keys
            - The model name used for completion
    """
    print("- requesting groq inference -")
    
    # Initialize the messages list with system prompt and initial assistant acknowledgment
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
        
    # Initialize Groq client and perform inference
    with Groq(api_key=api_keys.groq_api_key) as client:
        # Get available models and filter for Meta models with large context windows
        models = client.models.list()
        models = [model for model in models.data if "Meta" in model.owned_by and int(model.context_window) >= 32768]
        models = sorted(models, key=lambda x: x.created, reverse=True)   
        print(f"- Using model: {models[0].id} -")
        
        # Create chat completion request
        chat_completion = client.chat.completions.create(
            messages=messages,
            model = models[0].id,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        response_text = chat_completion.choices[0].message.content
    
    # Clean up client
    if client:
        del client
    
    # Process response for thinking steps if present
    if '<think>' in response_text:
        think_content = response_text[response_text.find("<think>") + 7:response_text.find("</think>")].strip()
        rest_content = response_text[response_text.find("</think>") + 8:].strip()
        
        if is_think_needed:
            response_text = {
                "think": think_content,
                "response" : rest_content
            }
            response_text, chat_completion.model
            return ToolCallResponseSchema(
                tool_used="Groq Inference",
                short_description_of_the_tool=f"Inference from Groq AI platform, model: {chat_completion.model}.",
                result=response_text
            ) 
        else:
            return ToolCallResponseSchema(
                tool_used="Groq Inference",
                short_description_of_the_tool=f"Inference from Groq AI platform, model: {chat_completion.model}.",
                result=rest_content
            ) 
    else:
        return ToolCallResponseSchema(
            tool_used="Groq Inference",
            short_description_of_the_tool=f"Inference from Groq AI platform, model: {chat_completion.model}.",
            result=response_text
        )    

tool_groq: Tool = Tool(
    name="groq_inference",
    func=groq_inference,
    description="Performs inference on a given message using specific parameters. Returns a tuple containing the result and reasoning.",
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The input message to process."
            },
            "temperature": {
                "type": "number",
                "description": "Controls randomness in output generation. Higher values produce more random outputs.",
                "default": 0.8
            },
            "max_completion_tokens": {
                "type": "integer",
                "description": "The maximum number of tokens to generate in the completion.",
                "default": 4096
            },
            "is_think_needed": {
                "type": "boolean",
                "description": "Indicates whether reasoning or 'thinking' is required for the task.",
                "default": True
            },
            "task_memory_messages": {
                "type": "array",
                "items": {
                    "type": "object"
                },
                "description": (
                    "A list of prior messages or context objects relevant to the task."
                ),
                "default": []
            }
        },
        "required": ["message"]
    }
)



if __name__ == "__main__":
    """
    Test function for the groq_inference method.
    Loads configuration, sets up a simple test case, and prints the response.
    """
    system_message = "You are a helpful AI assistant. You do no write anything unnecessary, reply concise and short results."
    message = "count one to five"
    
    api_key = api_keys.groq_api_key
    response_text, chat_completion_model = groq_inference(message=message, api_key=api_key, system_message=system_message)
    pprint.pprint(response_text)
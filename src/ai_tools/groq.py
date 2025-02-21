from asyncio.log import logger
import json
import os
from pathlib import Path
import pprint
from typing import Any
from groq import Groq

from src.ai_tools.tools_utils import process_chat_complettion_prompt
from src.utils.utils import load_config

def groq_inference(message: str, system_message: str, model: str, api_key: str, temperature: int = 0.8, max_completion_tokens: int = 2048, is_think_needed: bool = False, task_memory_messages: list[Any] = []) -> tuple[str, str]:
    print("- requesting groq inference")
    # client = Groq(api_key=api_key)
    
    messages = [
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": "Okay."},
            ]
    # process task_memory_messages 
    if len(task_memory_messages) > 0:  
        for mem_seg in task_memory_messages:
            
            if mem_seg.role is "user":
                messages.append({"role": "user", "content": mem_seg.content})
            elif mem_seg.role is not "assistant":
                temp = {
                    "role": mem_seg.role,
                    "type": mem_seg.type, 
                    "content": mem_seg.content,
                    "timestamp": mem_seg.timestamp
                }
                messages.append({"role": "assistant", "content": f"Memory segments - {mem_seg.timestamp}:\n {json.dumps(temp, indent=2)}"})
            elif mem_seg.role is "assistant":
                messages.append({"role": "assistant", "content": mem_seg.content})
    
    messages.append({"role": "user", "content": message})
    
    with Groq(api_key=api_key) as client:
        chat_completion = client.chat.completions.create(
            messages=,
            model = model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        response_text = chat_completion.choices[0].message.content
    
    if client:
        del client
    # split "think" text chunk
    if '<think>' in response_text:
        think_content = response_text[response_text.find("<think>") + 7:response_text.find("</think>")].strip()
        rest_content = response_text[response_text.find("</think>") + 8:].strip()
        # print("think texts: ", think_content)
        # print("reply: ", rest_content)
        if is_think_needed:
            response_text = {
                "think": think_content,
                "response" : rest_content
            }
            return response_text, chat_completion.model,
        else:
            return rest_content, chat_completion.model
    else:
        return response_text, chat_completion.model

# test
"""
def test_groq_inference():
    config = load_config()
    system_message = "You are a helpful AI assistant. You do no write anything unnecessary, reply concise and short results."
    message = "count one to five"
    model = config.get("groq_model_name")
    api_key = config.get("groq_api_key")
    response_text, chat_completion_model = groq_inference(message=message, model=model, api_key=api_key, system_message=system_message)
    
    
    
test_groq_inference()
"""
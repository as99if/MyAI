from asyncio.log import logger
import json
import os
from pathlib import Path
import pprint
from groq import Groq

from src.utils.utils import load_config

def groq_inference(message: str, system_message: str, model: str, api_key: str, temperature: int = 0.8, max_completion_tokens: int = 2048) -> tuple[str, str]:
    print("- requesting groq inference")
    # client = Groq(api_key=api_key)
    with Groq(api_key=api_key) as client:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": message},
            ],
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
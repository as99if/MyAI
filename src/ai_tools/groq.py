from asyncio.log import logger
import json
import os
from pathlib import Path
from groq import Groq

from src.utils.utils import load_config

def groq_inference(message: str, system_message: str, model: str = None, temperature: int = 0.8, max_completion_tokens: int = 4096) -> tuple[str, str]:
    config = load_config()
    if model is None:
        model = config.get("groq_model")
    
    client = Groq(api_key=config.get("groq_api_key"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": system_message},
            {"role": "user", "content": message},
        ],
        model = model,
        # model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=0.95,
        stream=False,
        stop=None,
    )

    response_text = chat_completion.choices[0].message.content
    print(response_text)
    return response_text, chat_completion.model

# groq_inference('count one to five', 'you are a helpful assistant which only can write numbers')
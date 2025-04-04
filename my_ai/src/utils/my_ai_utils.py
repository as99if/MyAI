from asyncio.log import logger
import json
from pathlib import Path
    
def load_prompt(path: str = "src/config/prompts/system_prompts.json"):
    prompt_path = Path(path)
    try:
        with open(prompt_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {prompt_path}")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config file")
        raise

def split_list(input_list, chunk_size):
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]


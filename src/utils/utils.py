from asyncio.log import logger
import json
from pathlib import Path


def load_config(path: str = "src/config.json"):
    config_path = Path(path)
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config file")
        raise
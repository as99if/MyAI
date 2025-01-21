import json
import json
import logging
import asyncio
from pathlib import Path

from src.core.my_ai import MyAIAssistant
from src.inference_engine.inference_engine import InferenceEngine
from src.memory.conversation_history_service import ConversationHistoryEngine


def load_config():
    config_path = Path("src/config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config file")
        raise


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
        # TODO: check for conversation backup and summarize if needed in a different thread
        conveversation_history_service = ConversationHistoryEngine(
            config=config)
        asyncio.run(conveversation_history_service.connect())
        logger.info("Connected to conversation history service")
        innference_engine = InferenceEngine(
            config=config, conversation_history_service=conveversation_history_service)
        assistant = MyAIAssistant(config=config, inference_engine=innference_engine,
                                  conversation_history_service=conveversation_history_service)
        logger.info("AI Assistant initialized successfully")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

from asyncio import subprocess
import json
import json
import logging
import asyncio
import multiprocessing
from pathlib import Path

from src.ai_configurations.app_configuration import AppConfiguration
from src.core.my_ai import MyAIAssistant
from src.inference_engine.inference_processor import InferenceProcessor
from src.inference_engine.inference_server import InferenceServer
from src.memory_processor.conversation_history_engine import ConversationHistoryEngine
from src.speech_engine.speech_service import ASREngine, TTSEngine
from src.memory_processor.conversation_summarizer import ConversationSummarizer
from src.utils.utils import load_config


logger = logging.getLogger(__name__)


def __run__():
    try:
        config = load_config("src/config.json")
        app_config = AppConfiguration(config=config)
        app_config.connect()
        app_config = app_config._get_all_data()
        logger.info("Configuration loaded successfully")
        
        asr_service = ASREngine()
        tts_service = TTSEngine()
        logger.info("Speeck engine loaded successfully")
        
        conveversation_history_engine = ConversationHistoryEngine(
            config=config)
        asyncio.run(conveversation_history_engine.connect())
        logger.info("Connected to conversation history service")
        
        innference_processor = InferenceProcessor(
            conversation_history_service=conveversation_history_engine)

        innference_processor._check_inference_server_health()
        # check inference server health
        logger.info("LLM initialized successfully")
        
        
        assistant = MyAIAssistant(innference_processor=innference_processor,
                                  conversation_history_service=conveversation_history_engine)
        logger.info("AI Assistant initialized successfully")

        # put this on exit
        # summariser = ConversationSummarizer(inference_engine=innference_engine, conversation_history_engine=conveversation_history_engine, app_config=app_config)
        # summariser.summarise_and_process_conversation()
        # concurrent
        # summariser.start_summarization_thread()
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise Exception(f"Error during initialization: {str(e)}")

if __name__ == "__main__":
    
    p = multiprocessing.Process(target=__run__, name="computer")
    p.start()
    p.join()
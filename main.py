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
# from src.memory_processor.conversation_history_engine import ConversationHistoryEngine
from src.speech_engine.speech_engine import SpeechEngine
# from src.memory_processor.conversation_summarizer import ConversationSummarizer
from src.utils.utils import load_config
from src.utils import gui_util

logger = logging.getLogger(__name__)


def __run__():
    try:
        config = load_config("src/config.json")
        app_config = AppConfiguration(config=config)
        # asyncio.run(app_config.connect())
        
        # app_config = asyncio.run(app_config._get_all_data())
        
        #logger.info("Configuration loaded successfully")
        
        speech_engine = SpeechEngine()
        logger.info("Speech engine loaded successfully")
        
        # conveversation_history_engine = ConversationHistoryEngine()
        # asyncio.run(conveversation_history_engine.connect())
        # logger.info("Connected to conversation history service")
        
        conveversation_history_engine = None
        inference_processor = InferenceProcessor()
        logger.info("LLM processor initiated")

        # check if inference server started or not
        server_running = asyncio.run(inference_processor._check_inference_server_health())
        # check inference server health
        
        
        if server_running:
            logger.info("LLM server running")
            my_ai = MyAIAssistant(inference_processor=inference_processor,
                                    conversation_history_engine=conveversation_history_engine,
                                    speech_engine=speech_engine)
            logger.info("AI Assistant initialized successfully")
            my_ai.run()
            gui_util.gui_interface = my_ai.gui_interface
            # most of the operational prints will be shown there
            logger.info("AI Assistant started successfully")
        else:
            print("LLM server not running")

            
            
        # put this on exit
        # summariser = ConversationSummarizer(inference_engine=innference_engine, conversation_history_engine=conveversation_history_engine, app_config=app_config)
        # summariser.summarise_and_process_conversation()
        # concurrent
        # summariser.start_summarization_thread()
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise Exception(f"Error during initialization: {str(e)}")

if __name__ == "__main__":

    computer = multiprocessing.Process(target=__run__, name="computer")
    computer.start()
    computer.join()
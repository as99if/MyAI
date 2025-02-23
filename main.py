from asyncio import subprocess
import json
import json
import logging
import asyncio
import multiprocessing
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from src.ai_configurations.app_configuration import AppConfiguration
from src.api.api import ChatAPI
from src.core.my_ai import MyAIAssistant
from src.core.my_ai_ui import MyAIUI
from src.inference_engine.inference_processor import InferenceProcessor
# from src.memory_processor.conversation_history_engine import ConversationHistoryEngine
from src.speech_engine.speech_engine import SpeechEngine
# from src.memory_processor.conversation_summarizer import ConversationSummarizer
from src.utils.utils import load_config
from src.utils import gui_util
import uvicorn
logger = logging.getLogger(__name__)


async def start_fastapi_server(app):
    config = uvicorn.Config(app, host="0.0.0.0", port=9999)
    server = uvicorn.Server(config)
    await server.serve()

async def my_ai_run(my_ai, app):
    if my_ai.if_gui_enabled:
        await asyncio.gather(
            my_ai.run(),
            start_fastapi_server(app)
        )
    else:
        await asyncio.gather(
            my_ai.run(),
            start_fastapi_server(app)   #, gui
        )

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
            
            app = FastAPI()

            @app.get("/")
            def read_root():
                return {"message": "My AI API!"}

            chat_api = ChatAPI(conversation_history_engine=conveversation_history_engine, inference_processor=inference_processor)
            app.include_router(chat_api.router)
            
            if config.get('api_only'):  
                # start my_ai api server only          
                uvicorn.run(app, host="0.0.0.0", port=9999)
                logger.info("My AI server API started successfully")
                
            else:
                # start FastAPI serrver 'app' parallaly of concurrently to the my_ai.run and gui
                asyncio.run(my_ai_run(my_ai, app))
                

            
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
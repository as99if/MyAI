from asyncio import subprocess
import json
import json
import logging
import asyncio
import multiprocessing
from pathlib import Path
from pyngrok import ngrok
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from src.core.api.api import MyAIChatAPI
from src.core.my_ai_assistant import MyAIAssistant
from src.inference_engine.inference_processor import InferenceProcessor
from src.memory_processor.conversation_history_engine import ConversationHistoryEngine
from src.speech_engine.speech_engine import SpeechEngine
from src.utils.utils import load_config

import uvicorn
logger = logging.getLogger(__name__)
class MyAI:
    def __init__(self, is_gui_enabled=False):
        self.is_gui_enabled = is_gui_enabled
        self.is_loading: bool = False
        self.my_ai_assistant = None
        self.config = None
        self.app_config = None
        self.speech_engine = None
        self.conveversation_history_engine = None
        self.inference_processor = None
    
    """
    async def start_fastapi_server(app):
        # tunnel
        config = load_config("src/config.json")
        if config.get('public_api'):
            public_url = ngrok.connect(9999).public_url
            print(f"Public URL: {public_url}") 
        config = uvicorn.Config(app, host="0.0.0.0", port=9999)
        server = uvicorn.Server(config)
        await server.serve()
    """

    async def run(self):
        self.is_loading: bool = True
        try:
            print('MyAI Initiating ...')
            logger.info("MyAI Initiating ...")
            self.config = load_config("src/config.json")
            
            self.speech_engine = SpeechEngine()
            logger.info("Speech engine loaded successfully")
            
            self.conveversation_history_engine = None
            # conveversation_history_engine = ConversationHistoryEngine()
            # asyncio.run(conveversation_history_engine.connect())
            # logger.info("Connected to conversation history service")
            
            
            self.inference_processor = InferenceProcessor()
            logger.info("LLM processor initiated")

            # check if inference server started or not
            self.server_running = await self.inference_processor._check_inference_server_health()
            # check inference server health
            
            
            if self.server_running:
                logger.info("LLM server running")
                app = FastAPI()

                @app.get("/")
                def read_root():
                    return {"message": "My AI API!"}

                self.chat_api = MyAIChatAPI(my_ai_assistant=self.my_ai_assistant)
                app.include_router(self.chat_api.router)
                
                if self.config.get('api_only'):  
                    # start my_ai api server only
                    if self.config.get('public_api'):
                        public_url = ngrok.connect(9999).public_url
                        print(f"Public URL: {public_url}")          
                    uvicorn.run(app, host="0.0.0.0", port=9999)
                    logger.info("My AI server API started successfully")
                    
                else:
                    self.my_ai_assistant = MyAIAssistant(inference_processor=self.inference_processor,
                                        conversation_history_engine=self.conveversation_history_engine,
                                        speech_engine=self.speech_engine)
                    logger.info("AI Assistant initialized successfully")
                    # TODO: starts MyAI with the FastAPI server
                    # start FastAPI serrver 'app' parallaly of concurrently to the my_ai.run and gui
                    # asyncio.run(my_ai_run(my_ai, app, config)) 
                    
            
                logger.info("AI Assistant started successfully")
            else:
                print("LLM server not running")
            
            if not self.is_gui_enabled:
                self.my_ai_assistant.run()
            
            self.is_loading: bool = False
            print('MyAI Initiation finished ...')
            logger.info("MyAI Initiation finished ...")

                
            # make conversation summarizer as a separet appliaction, like llama_cpp server
            # summariser = ConversationSummarizer(inference_engine=innference_engine, conversation_history_engine=conveversation_history_engine, app_config=app_config)
            # summariser.summarise_and_process_conversation()
            # concurrent
            # summariser.start_summarization_thread()
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            self.loading: bool = False
            raise Exception(f"Error during initialization: {str(e)}")
    
    def __run__(self):
        asyncio.run(self.run())
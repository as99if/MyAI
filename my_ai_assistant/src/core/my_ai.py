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

from src.core.api_server.api import MyAIChatAPI
from src.core.my_ai_assistant import MyAIAssistant
from src.inference_engine.inference_processor import InferenceProcessor
from my_ai.src.memory_processor.memory_processor import MemoryProcessor
from src.interface.speech_engine import SpeechEngine
from my_ai.src.utils.my_ai_utils import load_config
from src.utils.log_manager import LoggingManager

import uvicorn

class MyAI:
    def __init__(self, is_gui_enabled=False):
        self.logging_manager = LoggingManager()
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
            #logger.info("MyAI Initiating ...")
            self.logging_manager.add_message("MyAI Initiating ..", level="INFO", source="MyAI")
            self.config = load_config("src/config.json")
            
            self.speech_engine = SpeechEngine()
            self.logging_manager.add_message("Speech engine loaded successfully", level="INFO", source="MyAI")
            
            self.memory_processor = MemoryProcessor()
            asyncio.run(self.memory_processor.connect())
            self.logging_manager.add_message("Connected to memory services", level="INFO", source="MyAI")
            
            
            self.inference_processor = InferenceProcessor()
            self.logging_manager.add_message("Inference engine initiated", level="INFO", source="MyAI")

            # check if inference server started or not
            self.server_running = await self.inference_processor._check_inference_server_health()
            # check inference server health
            
            
            if self.server_running:
                #logger.info("LLM server running")
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
                    self.logging_manager.add_message("MyAI API server initiated", level="INFO", source="MyAI")
                    pass
                    
                else:
                    self.my_ai_assistant = MyAIAssistant(inference_processor=self.inference_processor,
                                        memory_processor=self.memory_processor,
                                        speech_engine=self.speech_engine)
                    #logger.info("AI Assistant initialized successfully")
                    # TODO: starts MyAI with the FastAPI server
                    # start FastAPI serrver 'app' parallaly of concurrently to the my_ai.run and gui
                    # asyncio.run(my_ai_run(my_ai, app, config)) 
                    
            
                self.logging_manager.add_message("System started successfully", level="INFO", source="MyAI")

            else:
                self.logging_manager.add_message("LLM server not running", level="ERROR", source="MyAI")

            
            if not self.is_gui_enabled:
                self.my_ai_assistant.run()
            
            self.is_loading: bool = False
            self.logging_manager.add_message("MyAI started successfully", level="INFO", source="MyAI")


                
            # make conversation summarizer as a separet appliaction, like llama_cpp server
            # summariser = ConversationSummarizer(inference_engine=innference_engine, conversation_history_processor=conversation_history_processor, app_config=app_config)
            # summariser.summarise_and_process_conversation()
            # concurrent
            # summariser.start_summarization_thread()
        except Exception as e:
            #logger.error(f"Error during initialization: {str(e)}")
            self.loading: bool = False
            raise Exception(f"Error during initialization: {str(e)}")
    
    def __run__(self):
        asyncio.run(self.run())
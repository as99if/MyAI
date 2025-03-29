"""
depreceted - this file is not used anymore

@author - Asif Ahmed - asif.shuvo2199@otlook.com

"""

import argparse
import asyncio
import multiprocessing
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import uvicorn
from llama_cpp import Llama

from src.utils.utils import load_config


class InferenceServer:
    def __init__(self):
        self.config = load_config()
        
        # for starting server from main of MyAI as a daemon
        # self.server_run = multiprocessing.Process(
        #     target=self._initialize_llm_server,
        #     name=name, 
        #     daemon=True
        # )
        # self.server_run.start()
        # self.server_run.join()

    def _initialize_llm_server(self):
        # for starting server from main of MyAI
        try:
            self.start_server()
            return True
        except Exception as e:
            print(f"Error initializing LLM server: {e}")
            raise e
            # return False

    def start_server(self, host: str = "0.0.0.0", port: int = 50001):
        name = "computer-inference-server"
        """Start the FastAPI server before starting myAI"""
        
        subprocess.run(["python", "-m", "llama_cpp.server", "--config_file", f"src/inference_engine/inference_server_config.json", "--port", "50001"])
        # see llama_cpp/server/settings.py for more options to use multiple models
        
        

    def _clear_inference_server(self):
        if self.inference_server:
            self.server_run.terminate()
            self.inference_server = None
            self.server_run = None


if __name__ == "__main__":
    inference_server = InferenceServer()
    inference_server.start_server()

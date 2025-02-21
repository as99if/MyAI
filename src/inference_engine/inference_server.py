"""
Llama CPP Inference Server
=========================

A FastAPI-based server providing OpenAI-compatible API endpoints for Llama language models.

Features:
- OpenAI-compatible REST API endpoints for text and chat completions
- Built on FastAPI for high performance async operations
- CORS-enabled for cross-origin requests
- Configurable model parameters (temperature, max_tokens)
- Model information endpoint


@author - Asif Ahmed - asif.shuvo2199@otlook.com

"""

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
    
    def _initialize_llm_server(self):
        try:
            self.inference_server = InferenceServer()
            self.inference_server.start_server()
            return True
        except Exception as e:
            print(f"Error initializing LLM server: {e}")
            raise e
            # return False
      
    def start_server(self, host: str = "0.0.0.0", port: int = 50001):
        """Start the FastAPI server"""
        llm_base_path = self.config.get("llm_base_path")
        llm = self.config.get("llm")
        # subprocess.run(["python", "-m", "llama_cpp.server", "--model", f"{llm_base_path}/{llm}", "--port", "50001"])
        subprocess.run(["python", "-m", "llama_cpp.server", "--config_file", f"src/inference_engine/inference_server_config.json", "--port", "50001"])
    
        self.server_run = multiprocessing.Process(target=self._initialize_llm_server, name="computer-inference-server")
        self.server_run.start()
        self.server_run.join()
    
    def _clear_inference_server(self):
        if self.inference_server:
            self.server_run.terminate()
            self.inference_server = None
            self.server_run = None

i = InferenceServer()
i.start_server()
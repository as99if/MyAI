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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import uvicorn
from llama_cpp import Llama

class ChatMessage(BaseModel):
    role: str
    content: str
    type: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256

class InferenceServer:
    def __init__(self, config: Dict[str, Any], inference_engine: Any):
        self.app = FastAPI(title="Llama CPP Python - API Server", description="OpenAI like API endpoints", version="0.1.0")
        self.config = config
        self.inference_engine = inference_engine
        self.client = inference_engine.client
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        @self.app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            try:
                response = self.client(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                return {
                    "id": "cmpl-xyz",
                    "object": "text_completion",
                    "created": None,
                    "model": request.model,
                    "choices": [{"text": response["choices"][0]["text"], "index": 0}]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            try:
                messages = [f"{m.role}: {m.content}" for m in request.messages]
                prompt = "\n".join(messages)
                
                response = self.client(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )
                
                return {
                    "id": "chatcmpl-xyz",
                    "object": "chat.completion",
                    "created": None,
                    "model": request.model,
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": response["choices"][0]["text"]
                        },
                        "index": 0
                    }]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/models")
        async def list_models():
            """Get information about loaded model"""
            try:
                model_path = self.config.get("model_path", "")
                return {
                    "object": "list", 
                    "data": [{
                        "id": model_path,
                        "object": "model",
                        "owned_by": "llamacpp",
                        "permissions": []
                    }]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
        @self.app.get("/models")
        async def list_models():
            """Get information about loaded model"""
            try:
                model_path = self.config.get("model_path", "")
                return {
                    "object": "list", 
                    "data": [{
                        "id": model_path,
                        "object": "model",
                        "owned_by": "llamacpp",
                        "permissions": []
                    }]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the FastAPI server"""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
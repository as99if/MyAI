from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.core.my_ai_assistant import MyAIAssistant
from src.core.api_server.data_models import MessageSegment, RequestMessageSegment


class MyAIChatAPI:
    def __init__(self, my_ai_assistant: MyAIAssistant):
        self.my_ai_assistant = my_ai_assistant
        self.router = APIRouter()
        self.router.post("/get_chat_response")(self.get_chat_response)

    async def get_chat_response(self, request: RequestMessageSegment) -> MessageSegment:
        
        try:
            response = await self.my_ai_assistant.create_chat_completion(
                message=request.message,
                is_api_request=True,
                is_audio_requested_in_api_response=request.is_audio_requested_in_api_response,
            )

            return {
                "response": response
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        

    
    # endpoint for getting conversation history
    # endpoint for chat response with audio
    # endpoint for chat response with audio
    # endpoint for chat response with video
    # endpoint for chat response with image

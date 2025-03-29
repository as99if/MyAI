from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.core.my_ai_assistant import MyAIAssistant

class ChatRequest(BaseModel):
    message: str


class MyAIChatAPI:
    def __init__(self, my_ai_assistant: MyAIAssistant):
        self.my_ai_assistant = my_ai_assistant
        self.router = APIRouter()
        self.router.post("/get_chat_response")(self.get_chat_response)

    async def get_chat_response(self, request: ChatRequest):
        try:
            response = await self.my_ai_assistant.create_chat_completion(
                message=request.message,
                is_api_request=True
            )
            
            return {
                "response": response
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_conversation_memory(self):
        pass
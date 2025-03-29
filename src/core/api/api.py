from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.core.my_ai_assistant import MyAIAssistant
from src.core.api.data_models import MessageSegment, RequestMessageSegment


class MyAIChatAPI:
    def __init__(self, my_ai_assistant: MyAIAssistant):
        self.my_ai_assistant = my_ai_assistant
        self.router = APIRouter()
        self.router.post("/get_chat_response")(self.get_chat_response)

    async def get_chat_response(self, request: RequestMessageSegment) -> MessageSegment:
        if not request.content is type(str):
            if request.content.type is "image_url":
                if_screenshot_or_image = True
                is_vision_enabled = True
            elif request.content.type is "audio":
                is_audio_in_api_request = True
            elif request.content.type is "video":
                is_vision_enabled = True
                if_video_file = False
        
        try:
            response = await self.my_ai_assistant.create_chat_completion(
                message=request,
                is_api_request=True,
                is_audio_requested_in_api_response=request.is_audio_requested_in_api_response,
                is_vision_enabled = is_vision_enabled,
                if_video_file = if_video_file,
                if_screenshot_or_image = if_screenshot_or_image,
                image_urls = request.image_urls,
                video_urls = request.video_urls,
                audio_urls = request.audio_urls,
                file_urls = request.file_urls,
                is_audio_in_api_request = is_audio_in_api_request,
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

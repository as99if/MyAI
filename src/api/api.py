from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatAPI:
    def __init__(self, conversation_history_engine, inference_processor):
        self.conversation_history_engine = conversation_history_engine
        self.inference_processor = inference_processor
        self.router = APIRouter()
        self.router.post("/get_chat_response")(self.get_chat_response)

    async def get_chat_response(self, request: ChatRequest):
        try:
            user_message = [
                {
                    "role": "user",
                    "content": request.message,
                    type: "user_message",
                    "timestamp": datetime.now().isoformat(),
                }
            ]
            recent_conversation = []
            if self.conversation_history_engine:
                try:
                    recent_conversation = (
                        await self.conversation_history_engine.get_recent_conversation()
                    )
                    recent_conversation = recent_conversation + user_message
                    self.conversation_history_engine.add_conversation(user_message)
                except Exception as e:
                    print(f"Error getting recent conversation: {e}")
                    raise e
            else:
                recent_conversation = recent_conversation + user_message

            response = await self.inference_processor.create_chat_completion(
                recent_conversation
            )

            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

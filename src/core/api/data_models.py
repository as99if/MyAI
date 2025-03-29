from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class MessageContent(BaseModel):
    type: str = "text" | "image_url" | "audio" | "video" | "file"
    content: Any  # "str" | {type: "image_url", "image_url"} | {type: "audio", "audio_url": "audio_url"} | {type: "video", "video_url":"video_url"} | {"type":"file", "file_url":"file_url"}


class MessageSegment(BaseModel):
    role: Optional[str] = "user" | "assistant" | "system"
    content: str | list[MessageContent] | None
    timestamp: datetime


class RequestMessageSegment(BaseModel):
    role: Optional[str] = "user" | "assistant" | "system"
    content: str | list[MessageContent] | None
    timestamp: datetime
    is_audio_requested_in_api_response: bool = False
    image_urls: Optional[list[str]] = None
    video_urls: Optional[list[str]] = None
    audio_url: Optional[list[str]] = None
    file_urls: Optional[list[str]] = None
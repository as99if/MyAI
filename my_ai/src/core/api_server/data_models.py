from datetime import datetime
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, HttpUrl
            
class Attachment(BaseModel):
    url: Union[HttpUrl, str]
    description: Optional[str] = None
    title: Optional[str] = None
    file_type: Optional[str] = None


class ContentSegment(BaseModel):
    type: Literal["text", "image_url", "audio", "video", "file"]
    text: Optional[str] = None
    image_url: Optional[Attachment] = None
    audio_url: Optional[Attachment] = None
    video_url: Optional[Attachment] = None
    file_url: Optional[Attachment] = None
    description: Optional[str] = None


class MessageContent(BaseModel):
    """
    MessageContent is a class that represents a message in the chat application.
    It contains the role of the message sender (user, assistant, or system),
    a timestamp indicating when the message was sent, and the content of the message.
    The content can be a list of strings or ContentSegment objects,
    which can include text, image URLs, audio URLs, video URLs, or file URLs.
    
    e.g.: 
    message = {
        "role": "user",
        "timestamp": "<timestamp>",
        "content": [
            "<message text>",
            {``
                "type": "text",
                "text": "<message text>",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "<image_url>",
                    "description": "<description>",
                    "title": "<text>",
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "<image_url>",
                    "description": "<description>",
                    "title": "<text>",
                },
            },
            {
                "type": "audio",
                "audio_url": {"url": "<audio_url>", "description": "<description>"},
            },
            {
                "type": "audio",
                "audio_url": {"url": "<audio_url>", "description": "<description>"},
            },
            {
                "type": "video",
                "video_url": {
                    "url": "<video_url>",
                    "description": "<description>",
                    "title": "<title>",
                },
            },
            {
                "type": "video",
                "video_url": {
                    "url": "<video_url>",
                    "description": "<description>",
                    "title": "<title>",
                },
            },
            {
                "type": "file",
                "file_url": {
                    "url": "<file_url>",
                    "description": "<description>",
                    "title": "<title>",
                    "file_type": "<file_type>",
                },
            },
            {
                "type": "file",
                "file_url": {
                    "url": "<file_url>",
                    "description": "<description>",
                    "title": "<title>",
                    "file_type": "<file_type>",
                },
            },
        ],
    }

    Args:
        BaseModel (_type_): _description_
    """
    role: Literal["user", "assistant", "system"]
    timestamp: datetime
    content: Union[str, List[Union[str, ContentSegment]]]
    type: Optional[str] = None
    unspoken_message: bool = False
    metadata: Optional[dict] = None


class RequestMessageSegment(BaseModel):
    message: MessageContent
    is_audio_requested_in_api_response: bool = False

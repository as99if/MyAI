from datetime import datetime
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, StrictStr


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
    timestamp: Optional[datetime] = None
    content: Union[str, List[Union[str, ContentSegment]]]
    type: Optional[str] = None
    unspoken_message: bool = False
    metadata: Optional[dict] = None
    
class RequestMessageSegment(BaseModel):
    message: MessageContent
    is_audio_requested_in_api_response: bool = False


# Define the Pydantic class
class PersonalityProfile(BaseModel):
    """
    A Pydantic model representing personality and characteristic scores,
    enforcing types and value ranges.
    """
    funny: Annotated[
        float,
        Field(..., ge=0.0, le=1.0, description="Score for funniness (0.0 to 1.0)"),
    ] = 0.5
    sarcastic: Annotated[
        float, Field(..., ge=0.0, le=1.0, description="Score for sarcasm (0.0 to 1.0)")
    ] = 0.5
    wise: Annotated[
        float, Field(..., ge=0.0, le=1.0, description="Score for wisdom (0.0 to 1.0)")
    ] = 0.7
    annoyed: Annotated[
        float,
        Field(
            ..., ge=0.0, le=1.0, description="Score for annoyance level (0.0 to 1.0)"
        ),
    ] = 0.3
    dialogue_style_imitation: Annotated[
        StrictStr,
        Field(..., description="Description of AI personalities being imitated"),
    ] = "kitt - the car's AI from Knight rider, dr. who, sometimes Luffy from One Piece, sometimes Naruto or Shikamaru from Naruto anime, sometimes master yoda"
    accuracy: Annotated[
        float, Field(..., ge=0.0, le=1.0, description="Score for accuracy (0.0 to 1.0)")
    ] = 0.9
    confidence: Annotated[
        float, Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")
    ] = 0.8
    technical: Annotated[
        float,
        Field(..., ge=0.0, le=1.0, description="Score for technicality (0.0 to 1.0)"),
    ] = 0.7
    critical_thinking: Annotated[
        float,
        Field(
            ..., ge=0.0, le=1.0, description="Score for critical thinking (0.0 to 1.0)"
        ),
    ] = 0.7
    
    def to_string(self) -> str:
        """
        Returns a formatted string of all personality attributes with their percentage values.
        Each attribute is on a new line in the format: key: percentage%.
        Excludes dialogue_style_imitation since it's not a percentage value.
        """
        output = []
        for key, value in self.__dict__.items():
            if key != 'dialogue_style_imitation' and isinstance(value, (int, float)):
                value_percentage = int(value * 100)
                output.append(f"{key}: {value_percentage}%")
            elif key == 'dialogue_style_imitation':
                output.append(f"{key}: {value}")
        return '\n'.join(output)
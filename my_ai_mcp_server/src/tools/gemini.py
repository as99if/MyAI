import base64
import os
from google import genai
from google.genai import types
from pydantic import BaseModel
from src.utils.log_manager import LoggingManager
from src.utils.schemas import ToolCallResponse
from src.config.config import api_keys
import pprint
from langchain.tools import Tool


def gemini_inference(prompt:str) -> ToolCallResponse:
    """
    Processes a given prompt and returns a response based on Gemini's inference capabilities. It can access current events with the added Google Search helper here for grounding information.
    Args:
        prompt (str): A string prompt for the Gemini model to generate a response.
    Returns:
        str: response from the Gemini model.
    """
    print("Asking GEMINIIIII")
    logging_manager = LoggingManager()
    logging_manager.add_message("Initiating - Gemini tool", level="INFO", source="Gemini")
    logging_manager.add_message(f"Prompt: {prompt}", level="INFO", source="Gemini")
    
    
    client = genai.Client(
        api_key=api_keys.gemini_api_key,
    )

    model = "gemini-2.5-pro-exp-03-25"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch())
    ]
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    # pprint.pprint(response.candidates[0].content.parts[0].text)
    response = ToolCallResponse(
        tool_used="Gemini",
        short_description_of_the_tool="Inference from Gemini AI model with grounding with google search.",
        result=response.candidates[0].content.parts[0].text
    )
    logging_manager.add_message(f"Response: {response}", level="INFO", source="Gemini")
    
    return response


tool_gemini: Tool = Tool(
    name="Gemini Inference",
    func=gemini_inference,
    description="Processes a given prompt and returns a response based on Gemini's inference capabilities.",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The input prompt to process and generate a response."
            }
        },
        "required": ["prompt"]
    }
)

if __name__ == "__main__":
    gemini_inference("What is the current weather in Tokyo right now?")


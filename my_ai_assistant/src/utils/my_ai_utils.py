
import json
import pprint
from typing import List

from src.utils.log_manager import LoggingManager
from src.core.api_server.data_models import MessageContent
    

logging_manager = LoggingManager()

def split_list(input_list, chunk_size):
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]


def format_messages(messages: List[MessageContent]) -> list:
        _messages = []
        # print("\n\n*******before formatting for ui******")
        # pprint.pprint(messages)
        for msg in messages:
            if type(msg.content) == str:
                _messages.append({"role": msg.role, "content": msg.content})
            elif type(msg.content[0]) == str:
                _messages.append({"role": msg.role, "content": msg.content[0]})
            else:
                if msg.content[0].type == "text":
                    _messages.append({"role": msg.role, "content": msg.content[0].text})
            # handle other type of contents too
        # print("\n\n*******after formatting for ui******")
        # pprint.pprint(_messages)
        return _messages


def serialize_message_content_list(data: List[MessageContent]) -> list[str]:
    list_of_dicts = []
    try:
        for msg in data:
            list_of_dicts.append(msg.model_dump_json())
    except Exception as e:
        logging_manager.add_message(f"Error in serializing list of MessageContent to list of dict: {e}", level="ERROR", source="src.utils.my_ai_utils")
    
    return list_of_dicts

def deserialize_message_content_list(data: list[str]) -> List[MessageContent]:
    try:
        message_contents = []
        for json_str in data:
            # Parse JSON string to dictionary
            message_dict = json.loads(json_str)
            # Create MessageContent object from dictionary
            message = MessageContent(**message_dict)
            message_contents.append(message)
        return message_contents
    except Exception as e:
        logging_manager.add_message(f"Error in deserializing list into list of MessageContent: {e}", level="ERROR", source="src.utils.my_ai_utils")
    
# check core.md


import os
from datetime import datetime
import sys


import gc


import signal
import pprint

import threading
from typing import Any, List, Tuple, Optional
from src.inference_engine.inference_processor import InferenceProcessor
from src.memory_processor.memory_processor import MemoryProcessor
from src.interface.speech_engine import SpeechEngine
from src.config.config import load_prompt
from src.config.config import load_config
from src.core.api_server.data_models import ContentSegment, MessageContent, PersonalityProfile
from src.agent.my_ai_agent import MyAIAgent
from src.utils.log_manager import LoggingManager

class MyAIAssistant:

    def __init__(
        self,
        inference_processor: InferenceProcessor,
        speech_engine: SpeechEngine = None,
        memory_processor: MemoryProcessor = None,
        mock_conversation_history: List[MessageContent] = [],
        my_ai_agent: MyAIAgent = None
    ):  
        
        self.logging_manager = LoggingManager()
        self.logging_manager.add_message("Initiating - MyAIAssistant", level="INFO", source="MyAIAssistant")
        self._persoality_profile: PersonalityProfile = PersonalityProfile()
        
        self.system_prompts = load_prompt()
        self.config = load_config()
        self.inference_processor: InferenceProcessor = inference_processor
        self.memory_processor: MemoryProcessor = memory_processor
        
        if my_ai_agent == None:
            self.my_ai_agent: MyAIAgent = MyAIAgent()
        else:
            self.my_ai_agent: MyAIAgent = my_ai_agent
        
        self.if_gui_enabled = self.config.get("gui_enabled", False)
        self.voice_reply_enabled = self.config.get("voice_reply_enabled", False)

        
        self.speech_engine: SpeechEngine = speech_engine
        if self.speech_engine is not None:
            self.tts_engine = speech_engine.tts_engine
            self.asr_engine = speech_engine.asr_engine

        self.mock_conversation_history = mock_conversation_history


        # Text tracking for interruption handling
        self.spoken_reply = ""
        self.remaining_reply = ""


        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)
        self.logging_manager.add_message("Initiated - MyAIAssistant", level="INFO", source="MyAIAssistant")
    

    
    async def _add_messages_to_history(self, messages: List[MessageContent]) -> None:
        if self.memory_processor:
            try:
                await self.memory_processor.add_conversation(messages)
                self.logging_manager.add_message("Added message to conversation history", level="INFO", source="MyAIAssistant")
            except Exception as e:
                self.logging_manager.add_message(f"Error processing recent conversation: {e}", level="ERROR", source="MyAIAssistant")
                raise e
        return
    
        
    def process_input(self, sender: str="User", text: str="", content=None, frames: Any=None, image: Any=None, file: Any=None):
        """
        Handle user input from the UI and add it to the conversation.
        Args:
            sender (str): The sender of the message ("User" or "AI").
            content ...
        """

        # text only for now
        try:
            content = ContentSegment(text=text, type="text") if content is None else content
            message_content = MessageContent(
                role="user" if sender == "User" else "assistant",
                content=[content],
                timestamp=datetime.now().isoformat(),
                type="user_message" if sender == "User" else "computer_response"
            )
            return message_content
        except Exception as e:
            self.logging_manager.add_message(f"Error processing input: {e}", level="ERROR", source="MyAIAssistant")
            return None
    
    async def process_and_create_chat_generation(
        self,
        message: MessageContent,
        is_api_request: bool = False,
        is_audio_requested_in_api_response: bool = False,
        if_vision_inference: bool = False,
        if_camera_feed: bool = False,
        is_tool_call_permitted: bool = False,
        is_thinking_process_permitted: bool = False,
    ) -> Tuple[List[MessageContent], List[MessageContent]]:
        """
        Process the message and create a chat generation.
        It proceed processes with self-reflection, thinking and planning ahead of replying.
        And also determines if agent tool call is needed or not.
        After tool call (multiple excecutions) is proceeds with post reflection.
        * This will be used for API server and memory processor. Not for GUI directly.
        * GUI will call this method and then handle the TTS streaming separately.
        Args:
            message (MessageContent): The message content to process
            is_api_request (bool, optional): Flag for API request. Defaults to False.
            is_audio_requested_in_api_response (bool, optional): Flag for audio in API response. Defaults to False.
            if_vision_inference (bool, optional): Flag for vision model. Defaults to False.
            if_camera_feed (bool, optional): Flag for camera feed. Defaults to False.
        Returns:
            MessageContent: The processed message content
        """
        ## TODO: tool call not working from ui
        
        self.logging_manager.add_message("Processing message for chat generation", level="INFO", source="MyAIAssistant")
        
        # Initialize response and handle camera feed
        response: MessageContent = None
        if if_camera_feed:
            if_vision_inference = True

        # Set message type
        message.type = "user_message"
        message.unspoken_message = False
        
        # ensure that message is at index 0 of the content list, or it is just str and no content list
        _user_message_text: str = ""
        if type(message.content) == str:
            _user_message_text = message.content
        elif type(message.content[0]) == str:
            _user_message_text = message.content[0]
        elif message.content[0].type == "text":
            _user_message_text = message.content[0].text
        
        # Initialize conversation history either from mock or empty list
        # recent_conversation: List[MessageContent] = (
        #     self.mock_conversation_history
        #     if not self.memory_processor
        #     else []
        # )
        recent_conversation = []
        responses = []
        
        
        # handle self-reflection with flag and instruction
        _message: MessageContent = None
        
        _schema = None
        _message = message
        

        # arrange system messages to be placed at the start of the conversation recent_conversation
        persoality_profile: str = self._persoality_profile.to_string()
        # later on an agent will be able to change these persoality_profile variables
        system_messages = [
            MessageContent(
                role="user", # / "system" (based on model)
                content=f"System Prompt for chat assistant:\n{self.system_prompts['chatbot_system_prompt']}" + "\n"
                    + self.system_prompts["error_handling_instruction"] + "\n"
                    + self.system_prompts["fallback_response_instruction"],
                type="system_message",
                unspoken_message=True,
                timestamp=datetime.now().isoformat(),
            ),
            MessageContent(
                role="assistant",
                content=f"Okay.",
                type="computer_response",
                unspoken_message=True,
                timestamp=datetime.now().isoformat(),
            ),
            MessageContent(
                role="user", # / "system" (based on model)
                content=f"{self.system_prompts['personlity_prompt']}\nPersonality profile:\n{persoality_profile}\n{self.system_prompts['chatbot_guidelines']}",
                type="system_message",
                unspoken_message=True,
                timestamp=datetime.now().isoformat(),
            ),
            MessageContent(
                role="assistant",
                content=f"Personality Profile Settings Initialized.",
                type="computer_response",
                unspoken_message=True,
                timestamp=datetime.now().isoformat(),
            ),
        ]

        
        # Handle conversation history if engine exists
        if self.memory_processor:
            try:
                recent_conversations = await self.memory_processor.get_recent_conversations()
                self.logging_manager.add_message("Fetched recent conversation history", level="INFO", source="MyAIAssistant")
                if recent_conversations != None and len(recent_conversations) > 0:
                    # Add system messages and user message to conversation
                    recent_conversations = (
                            system_messages
                            + recent_conversations
                            + [_message]
                    )
                else:
                    recent_conversations = (
                            system_messages
                            + [_message]
                    )
                # Add user message to history
                await self._add_messages_to_history(
                    [_message]
                )

            except Exception as e:
                self.logging_manager.add_message(f"Error processing recent conversation: {e}", level="ERROR", source="MyAIAssistant")
                raise e
        else:
            # If no history engine, use local conversation
            recent_conversations = (
                system_messages
                + recent_conversation
                + [_message]
            )

        self.logging_manager.add_message("Formatted system instruction, context messages and user message", level="INFO", source="MyAIAssistant")
        
        # break if no inference processor initialised
        if self.inference_processor is None:
                        
            self.logging_manager.add_message("Inference processor is not initialized", level="ERROR", source="MyAIAssistant")
            return [MessageContent(
                role="assistant",
                content="Infernce processor is not initialized.",
                timestamp=datetime.now().isoformat(),
                type="computer_response"
            )], recent_conversations
        
        if is_tool_call_permitted and self.my_ai_agent.connected:
            pass
            ## do all these in my_ai_agent
            """
            # Generate initial self-reflection response
            self.logging_manager.add_message("Thinking ...", lsevel="INFO", source="MyAIAssistant")
            
            response = await self.inference_processor.create_chat_completion(
                messages=recent_conversations,
                schema=_schema
            )
            # check response.content[0] type if SelfReflection then proceed or retry
            response.type = "self_reflection"
            response.unspoken_message=True
            self.logging_manager.add_message("Thinking process completed", level="INFO", source="MyAIAssistant")

            # Add AI response to conversation history

            await self._add_messages_to_history([response])
            recent_conversations = recent_conversations + [response]

            # Handle tool calls
            try:
                # validate response for self_reflection
                self_reflection_validated = False
                if is_thinking_process_permitted:
                    self_reflection_validated = SelfReflection.model_validate_json(response.content[0])
                if self_reflection_validated:
                    self.logging_manager.add_message("Initiating - Agent", level="INFO", source="MyAIAssistant")
                        
                    # TODO: check the response properly.. pass the "content" : SelfReflection class only
                    agent_response = self.my_ai_agent.execute(user_query=_user_message_text, self_reflection=response.content[0])
                        
                    recent_conversations = recent_conversations + [agent_response]
                    await self._add_messages_to_history([agent_response])

                    # get another short response after this
                    
            except Exception as e:
                self.logging_manager.add_message(f"Error in replying - {e}", level="ERROR", source="MyAIAssistant")
                raise e
            """
            
        else:
            self.logging_manager.add_message("No Tool call initiated.", level="INFO", source="MyAIAssistant")
            self.logging_manager.add_message("Replying.", level="INFO", source="MyAIAssistant")
            response = await self.inference_processor.create_chat_completion(
                        messages=recent_conversations,
                        if_vision_inference=if_vision_inference,
                        if_camera_feed=if_camera_feed
            )

            self.logging_manager.add_message("Replied.", level="INFO", source="MyAIAssistant")
            
        if self.memory_processor:
            try:
                await self.memory_processor.add_conversation([response])
            except Exception as e:
                self.logging_manager.add_message(f"Error processing recent conversation: {e}", level="ERROR", source="MyAIAssistant")

        responses.append(response)
        recent_conversations = recent_conversations + [response]

        # TODO: multiple inference, self-reflection, tools, etc can be handled here before returning final response
        

        if response is not None:
            reply = self.post_process_response_for_speech_engine(response)
            # trigger tts engine to interface output as audio and/or text
            # self.speech_engine.speak(
            #     text=reply if reply is not None else "No reply generated.",
            #     stream=False, 
            #     visualize=True, 
            #     mute=False
            # )
            if self.speech_engine:
               # Run TTS in background so UI loop (pygame) is not blocked
               threading.Thread(
                   target=self.speech_engine.speak,
                   kwargs=dict(
                       text=reply if reply else "No reply generated.",
                       stream=True,
                       visualize=True,
                       mute=False
                   ),
                   daemon=True
               ).start()
            # reply (spoken and unspoken) display will be handled from the TTS engine callback in the future
            
        
        return responses, recent_conversations

    def post_process_response_for_speech_engine(self, message: MessageContent | str) -> str:
        """
        MessageContent to text for TTSEngine. (for now)

        Args:
            response (MessageContent or Any): reply from inference processor
        """
        print("to post_process_response_for_speech_engine")
        print(message)
        reply: str = ""
        if type(message) == str:
            reply = message
        elif type(message) == MessageContent:
            if type(message.content) == str:
                reply = message.content
            elif message.content.type == "text" and type(message.content) == str:
                reply = message.content[0]
            elif message.content[0].type == "text" and type(message.content[0]) == str:
                reply = message.content[0].text
        print("post_processed_response_for_speech_engine")
        print(reply)
        return reply


    def close(self, signum=None, frame=None):
        """
        Handles graceful shutdown of the assistant, cleaning up resources and closing connections.

        This method:
        1. Stops the audio interruption monitoring thread
        2. Closes PyAudio resources
        3. Saves any pending conversation history
        4. Closes speech engine connections
        5. Closes inference processor connections

        Args:
            signum: Signal number (optional, for signal handler compatibility)
            frame: Current stack frame (optional, for signal handler compatibility)
        """
        try:
            
            self.logging_manager.add_message("Initiating - MyAIAssistant shutdown", level="INFO", source="MyAIAssistant")

            # Close speech engine connections
            if self.speech_engine is not None:
                try:
                    self.speech_engine.close()
                except Exception as e:
                    self.logging_manager.add_message(f"Error closing speech engine: {e}", level="ERROR", source="MyAIAssistant")
            # Close inference processor connections
            if hasattr(self, "inference_processor"):
                try:
                    self.inference_processor.close()
                except Exception as e:
                    self.logging_manager.add_message(f"Error closing inference processor: {e}", level="ERROR", source="MyAIAssistant")  

            # Force garbage collection
            gc.collect()

            self.logging_manager.add_message("Shutdown complete. Goodbye!", level="INFO", source="MyAIAssistant")

            # Exit cleanly
            if signum is not None:
                sys.exit(0)

        except Exception as e:
            self.logging_manager.add_message(f"Error during shutdown: {e}", level="ERROR", source="MyAIAssistant")
            # Force exit if unable to clean up
            sys.exit(1)
        
# if __name__ == "__main__":
#     # Example usage
#     # Initialize the assistant with necessary components
#     inference_processor = InferenceProcessor()
#     my_ai_agent = MyAIAgent()
#     my_ai_assistant = MyAIAssistant(
#         inference_processor=inference_processor, my_ai_agent=my_ai_agent
#     )
#     response, conversation = asyncio.run(
#        my_ai_assistant.process_and_create_chat_generation(
#            message=MessageContent(
            #     role="user",
            #     content="Hello, can you tell me the current weather in Stuttgart?",
            #     timestamp=datetime.now().isoformat(),
            #     type="user_message",
            # )
#        )
#    )
    """print("recent_conversation")
    pprint.pprint(conversation, indent=4)
    print("response")
    pprint.pprint(response, indent=4)"""

    """
    async def _run_withput_gui(self, is_test: bool = False):
        

        if is_test:
            self.test_voice_reply()
            return

        print("Listening... (Hold space to record audio message)")
        print("Listening... (Hold something else to record video feed)")
        while True:
            message = self.listen()
            if message is None:
                continue

            print("Transcription:", message)

            # LLM chat completion
            # Prepare context messages

            self.process_chat_generation(message)

            if "clean and shutdown" in message.lower():
                print("Exiting...")
                self.exit_gracefully()
                break

    def run(self):

        if self.if_gui_enabled:
            pass
            # onlys start the MyAI assistant api server
        if not self.if_gui_enabled:
            try:
                # Create new event loop for the thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run the coroutine
                loop.run_until_complete(self._run_withput_gui())
            except Exception as e:
                print(f"Error in run: {e}")
            finally:
                loop.close()

    def test_voice_reply(self):
        text = "This is a test message, or text. Whatever.."
        self.voice_reply(text=text)
    """

# TODO: conversation history
# stt bad
# tts noisy

import asyncio
import json
import os
from datetime import datetime
import sys
import pyaudio
import wave
import numpy as np
import queue
import time
import gc

# import torch
# from pydub import AudioSegment
import math
import io
import keyboard  # Add this import
import gc
import threading

import time
import signal
import pprint
import numpy as np

from typing import Any, List, Tuple, Optional

from src.inference_engine.inference_processor import InferenceProcessor
from src.memory_processor.memory_processor import MemoryProcessor
from src.speech_engine.speech_engine import SpeechEngine
from src.config.config import load_prompt
from src.config.config import load_config
from src.core.api_server.data_models import ContentSegment, MessageContent, PersonalityProfile
from src.agent.my_ai_agent import MyAIAgent
from src.utils.log_manager import LoggingManager
from src.core.schemas import SelfReflection, ThinkingSchema

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

        # PyAudio configuration for audio detection
        self.CHUNK = 2048  # 1024, 4096
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1  # mono
        self.RATE = 44100
        self.WAVE_OUTPUT_FILENAME = "src/speech_engine/temp_audio.flac"
        self.pyaudio = pyaudio.PyAudio()

        # Audio detection parameters
        self.THRESHOLD = 500
        self.interruption = False

        # Text tracking for interruption handling
        self.spoken_reply = ""
        self.remaining_reply = ""

        # Average speaking rate (characters per second)
        self.CHARS_PER_SECOND = 15

        # Start the interruption monitoring thread
        self.monitor_audio_interruption_thread = threading.Thread(
            target=self.monitor_audio_interruption,
            name="monitor_audio_interruption",
            daemon=True,
        )
        # Interrupting monitoring
        self.monitor_audio_interruption_thread.start()

        # self.gui_visualizer = SpectrogramWidget()

        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.logging_manager.add_message("Initiated - MyAIAssistant", level="INFO", source="MyAIAssistant")
        

    def split_text_into_chunks(self, text, chunk_size=100) -> list:
        """
        Splits text into smaller chunks for progressive text-to-speech processing.

        Args:
            text (str): The input text to split
            chunk_size (int, optional): Maximum characters per chunk. Defaults to 100.

        Returns:
            list: List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            if current_size + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def listen(self, duration=5) -> Optional[str]:
        """
        Records and transcribes user speech input.

        Args:
            duration (int, optional): Maximum recording duration in seconds. Defaults to 5.

        Returns:
            Optional[str]: Transcribed text or None if transcription fails
        """
        if self.speech_engine != None:
            audio_queue = queue.Queue()
            frames = []

            def audio_callback(in_data, frame_count, time_info, status):
                audio_queue.put(in_data)
                return (None, pyaudio.paContinue)

            stream = None
            try:
                # while condition : key pressed
                while keyboard.is_pressed("space"):
                    if stream is None:
                        stream = self.pyaudio.open(
                            format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK,
                            stream_callback=audio_callback,
                        )
                    if not audio_queue.empty():
                        data = audio_queue.get()
                        frames.append(data)
            finally:
                if stream:
                    stream.stop_stream()
                    stream.close()

            if len(frames) > 0:

                # Save the recorded audio to a temporary WAV file
                with wave.open(self.WAVE_OUTPUT_FILENAME, "wb") as wf:
                    # print("Processing audio...")
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.pyaudio.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(b"".join(frames))

                try:
                    # Use Whisper CPP for transcription

                    result = self.asr_engine.engine.transcribe(
                        media=self.WAVE_OUTPUT_FILENAME
                    )

                    # Clean up temporary file
                    if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                        os.remove(self.WAVE_OUTPUT_FILENAME)

                    return result[0].text if result else None

                except Exception as e:
                    self.logging_manager.add_message(f"Transcription error: {e}", level='INFO', source='MyAIAssistant')
                    if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                        os.remove(self.WAVE_OUTPUT_FILENAME)
                    return None
        else:
            self.logging_manager.add_message(f"Speech engine is not initialized.", level='INFO', source='MyAIAssistant')

            return None

    def voice_reply(self, text, is_api_request: bool = False) -> Tuple[str, str]:
        """
        Converts text to speech with interruption handling.

        Args:
            text (str): Text to be spoken

        Returns:
            Tuple[str, str]: (spoken_reply, remaining_reply)
            - spoken_reply: The portion of text that was successfully spoken
            - remaining_reply: Unspoken text if interrupted
            - OR
            - audio file or stream if is_api_request is True
        """
        if self.speech_engine != None:
            if is_api_request:
                try:

                    # # Create in-memory file-like object
                    audio_buffer = io.BytesIO()

                    # # Save as WAV file in memory
                    with wave.open(audio_buffer, "wb") as wav_file:
                        self.tts_engine.synthesize_wav(text, wav_file=wav_file, syn_config=self.tts_engine.engine_config)
                    
                    # # Get the audio data as bytes
                    audio_data = audio_buffer.getvalue()
                    audio_buffer.close()
                    return audio_data

                except Exception as e:
                    self.logging_manager.add_message(f"Error generating audio for API response: {e}", level='ERROR', source='MyAIAssistant')


                    raise e

            if not is_api_request:
                try:
                    # Reset flags and text tracking
                    self.interruption = False
                    self.spoken_reply = ""
                    self.remaining_reply = text

                    # Split text into manageable chunks
                    chunks = self.split_text_into_chunks(text)

                    # detect key press for interruption handling

                    # Process each chunk of text for TTS synthesis and playback
                    for i, chunk in enumerate(chunks):
                        if self.interruption:  # Check if user interrupted by speaking
                            self.logging_manager.add_message("Interruption detected!", level="INFO", source="MyAIAssistant")
                            self.remaining_reply = " ".join(chunks[i:])
                            break

                        samples, self.kokoro_sample_rate = (
                            self.tts_engine.engine.create(
                                [chunk], voice="am_adam", speed=0.92, lang="en-us"
                            )
                        )
                        # # Save as WAV file in memory
                        with wave.open(audio_buffer, "wb") as wav_file:
                            self.tts_engine.synthesize_wav(text, wav_file=wav_file, syn_config=self.tts_engine.engine_config)
                    
                        # print("Playing audio...")
                        
                        # Convert numpy array to bytes
                        audio_data = samples.astype(np.int16).tobytes()
                        
                        # Create PyAudio stream for playback
                        stream = self.pyaudio.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=self.kokoro_sample_rate,
                            output=True
                        )

                        # Play the audio and monitor interruptions
                        start_time = time.time()
                        chunk_size = 1024  # Adjust this value based on your needs
                        
                        for i in range(0, len(audio_data), chunk_size):
                            if self.interruption:
                                elapsed_time = time.time() - start_time
                                spoken_chars = int(elapsed_time * self.CHARS_PER_SECOND)
                                current_chunk_spoken = chunk[:spoken_chars]
                                self.spoken_reply += current_chunk_spoken + " "
                                break
                            
                            stream.write(audio_data[i:i + chunk_size])
                            time.sleep(0.01)  # Small delay to prevent busy-waiting

                        # Clean up stream
                        stream.stop_stream()
                        stream.close()

                        # If no interruption occurred during this chunk playback
                        if not self.interruption:
                            self.spoken_reply += chunk + " "

                    print("\nSpoken text:", self.spoken_reply)
                    self.remaining_reply = f"(Voice reply interrupted, remaining unsaid reply)\n{self.remaining_reply}"
                    print("\nRemaining text:", self.remaining_reply)

                    return self.spoken_reply, self.remaining_reply
                except Exception as e:
                    self.logging_manager.add_message(f"Error in text-to-speech: {e}", level='ERROR', source='MyAIAssistant')
        else:
            self.logging_manager.add_message(f"Speech engine is not initialized.", level='INFO', source='MyAIAssistant')

            return None, None

    def monitor_audio_interruption(self):
        """
        Continuously monitors for user interruptions via keyboard input.
        Runs in a separate thread to enable real-time interruption detection.
        """
        if self.speech_engine != None:
            while True:
                if keyboard.is_pressed("space"):
                    self.interruption = True
                else:
                    self.interruption = False
                time.sleep(0.1)  # Prevent busy-waiting
        else:
            pass

    def exit_gracefully(self, signum=None, frame=None):
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
            print("\nInitiating graceful shutdown...")
            self.logging_manager.add_message("Initiating - MyAIAssistant graceful shutdown", level="INFO", source="MyAIAssistant")
        

            # Stop the audio interruption monitoring thread
            if hasattr(self, "monitor_audio_interruption_thread"):
                self.interruption = True  # Signal thread to stop
                if self.monitor_audio_interruption_thread.is_alive():
                    self.monitor_audio_interruption_thread.join(timeout=2.0)

            # Clean up PyAudio resources
            if hasattr(self, "pyaudio"):
                try:
                    self.pyaudio.terminate()
                except Exception as e:
                    print(f"Error terminating PyAudio: {e}")

            # Clean up temporary audio files
            if hasattr(self, "WAVE_OUTPUT_FILENAME"):
                if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                    try:
                        os.remove(self.WAVE_OUTPUT_FILENAME)
                    except Exception as e:
                        print(f"Error removing temporary audio file: {e}")

            # Close speech engine connections
            if hasattr(self, "speech_engine") and self.speech_engine:
                try:
                    if self.tts_engine:
                        self.tts_engine.close()
                    if self.asr_engine:
                        self.asr_engine.close()
                except Exception as e:
                    print(f"Error closing speech engines: {e}")

            # Close inference processor connections
            if hasattr(self, "inference_processor"):
                try:
                    self.inference_processor.close()
                except Exception as e:
                    print(f"Error closing inference processor: {e}")

            # Force garbage collection
            gc.collect()

            print("Shutdown complete. Goodbye!")

            # Exit cleanly
            if signum is not None:
                sys.exit(0)

        except Exception as e:
            print(f"Error during shutdown: {e}")
            self.logging_manager.add_message(f"Error during shutdown: {e}", level="ERROR", source="MyAIAssistant")
            # Force exit if unable to clean up
            sys.exit(1)
            
    def close(self):
        """
        Close the assistant and clean up resources.
        """
        self.exit_gracefully()
        
    async def _add_messages_to_history(self, messages: List[MessageContent]) -> None:
        if self.memory_processor:
            try:
                await self.memory_processor.add_conversation(messages)
                self.logging_manager.add_message("Added message to conversation history", level="INFO", source="MyAIAssistant")
            except Exception as e:
                print(f"Error processing recent conversation: {e}")
                raise e
        return

    async def process_and_create_chat_generation(
        self,
        message: MessageContent,
        is_api_request: bool = False,
        is_audio_requested_in_api_response: bool = False,
        if_vision_inference: bool = False,
        if_camera_feed: bool = False,
        is_tool_call_permitted: bool = True,
        is_thinking_process_permitted: bool = False,
    ) -> Tuple[MessageContent, List[MessageContent]]:
        """
        Process the message and create a chat generation.
        It proceed processes with self-reflection, thinking and planning ahead of replying.
        And also determines if agent tool call is needed or not.
        After tool call (multiple excecutions) is proceeds with post reflection.
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
                print(f"Error processing recent conversation: {e}")
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
            return MessageContent(
                role="assistant",
                content="Infernce processor is not initialized.",
                timestamp=datetime.now().isoformat(),
                type="computer_response"
            ), recent_conversations
        
        if is_tool_call_permitted:
            pass
            ## do all these in my_ai_agent
            """
            # Generate initial self-reflection response
            self.logging_manager.add_message("Thinking ...", level="INFO", source="MyAIAssistant")
            
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
            
            

        # Deliver response
        # Handle API requests
        """if is_api_request:
            if not is_audio_requested_in_api_response:
                # Add response to history before returning
                await self._add_messages_to_history([response])
                recent_conversations = recent_conversations + [response]
                return response, recent_conversations

            # Handle audio in API response
            text_response = response.content[0]
            if response.content[0].type == "text":
                text_response = response.content[0].content
            if self.speech_engine != None and response.type != "unspoken":
                audio_reply = self.voice_reply(
                    text_response, is_audio_requested_in_api_response
                )
                # reponse - audio and text
                response_content = [
                    ContentSegment(
                        type="text",
                        content=text_response,
                        description="text reply",
                    ),
                    ContentSegment(
                        type="audio",
                        audio_url=audio_reply,
                        description="audio of the text reply",
                    ),
                ]
                response = MessageContent(
                    user="assistant",
                    content=response_content,
                    timestamp=datetime.now().isoformat(),
                    type="computer_response_with_audio",
                )
                # Add response with audio to history before returning
                """

        # Handle voice reply for non-API requests
        """if (
            not is_api_request
            and self.voice_reply_enabled
            and self.speech_engine != None
        ):

            text_response = response.content[0]
            if response.content[0].type == "text":
                text_response = response.content[0].content

            spoken_reply, unspoken_reply = self.voice_reply(
                text_response,
                is_audio_requested_in_api_response,
            )
            reply = (
                f"{spoken_reply}"
                if self.remaining_reply != ""
                else f"{spoken_reply}\n[- Interrupted, Remaining Unspoken Reply: {unspoken_reply} -]"
            )

            response = MessageContent(
                role="assistant",
                message=ContentSegment(type="text", content=reply),
                timestamp=datetime.now().isoformat(),
                type="computer_message",
            )
            # Add voice response to history before returning
        """
        if self.memory_processor:
            try:
                await self.memory_processor.add_conversation([response])
            except Exception as e:
                self.logging_manager.add_message(f"Error processing recent conversation: {e}", level="ERROR", source="MyAIAssistant")

        recent_conversations = recent_conversations + [response]
        return response, recent_conversations

if __name__ == "__main__":
    # Example usage
    # Initialize the assistant with necessary components
    inference_processor = InferenceProcessor()
    my_ai_agent = MyAIAgent()
    my_ai_assistant = MyAIAssistant(
        inference_processor=inference_processor, my_ai_agent=my_ai_agent
    )
    response, conversation = asyncio.run(
        my_ai_assistant.process_and_create_chat_generation(
            message=MessageContent(
                role="user",
                content="Hello, can you tell me the current weather in Stuttgart?",
                timestamp=datetime.now().isoformat(),
                type="user_message",
            )
        )
    )
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

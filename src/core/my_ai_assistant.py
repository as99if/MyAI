# TODO: conversation history
# stt bad
# tts noisy

import asyncio
import json
import os
from datetime import datetime
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
import simpleaudio
import time
import signal
import pprint
import numpy as np

from typing import Any, Tuple, Optional

from src.utils.utils import load_config
from src.core.api.data_models import MessageSegment


class MyAIAssistant:

    def __init__(
        self,
        inference_processor,
        speech_engine,
        conversation_history_engine=None,
    ):
        self.computer = None
        self.config = load_config()
        self.inference_processor = inference_processor
        self.conversation_history_engine = conversation_history_engine
        self.if_gui_enabled = self.config.get("gui_enabled", False)
        self.voice_reply_enabled = self.config.get("voice_reply_enabled", False)

        self.tts_engine = speech_engine.tts_engine
        self.asr_engine = speech_engine.asr_engine

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
        audio_queue = queue.Queue()
        frames = []

        def audio_callback(in_data, frame_count, time_info, status):
            audio_queue.put(in_data)
            return (None, pyaudio.paContinue)

        stream = None
        try:
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

        # num_silent_frames = int(self.RATE * 1 / self.CHUNK)
        # silent_frame = b'\x00' * self.CHUNK * self.pyaudio.get_sample_size(self.FORMAT)
        if len(frames) > 0:
            # Append silent frames
            # for _ in range(num_silent_frames):
            #    frames.append(silent_frame)

            # Save the recorded audio to a temporary WAV file
            with wave.open(self.WAVE_OUTPUT_FILENAME, "wb") as wf:
                print("Processing audio...")
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
                print(f"Transcription error: {e}")
                if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                    os.remove(self.WAVE_OUTPUT_FILENAME)
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
        if is_api_request:
            pass
            # return an audio file or stream - tts with kokoro
        
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
                        print("Interruption detected!")
                        self.remaining_reply = " ".join(chunks[i:])
                        break

                    samples, self.kokoro_sample_rate = self.tts_engine.engine.create(
                        [chunk], voice="am_adam", speed=0.92, lang="en-us"
                    )
                    print("Playing audio...")
                    wave_obj = simpleaudio.WaveObject(
                        samples,
                        num_channels=1,
                        bytes_per_sample=2,
                        sample_rate=self.kokoro_sample_rate,
                    )

                    # Play the generated audio and monitor interruptions during playback
                    play_obj = wave_obj.play()
                    start_time = time.time()

                    while play_obj.is_playing():
                        # Check for user interruption
                        if self.interruption:
                            elapsed_time = time.time() - start_time
                            spoken_chars = int(elapsed_time * self.CHARS_PER_SECOND)
                            current_chunk_spoken = chunk[:spoken_chars]
                            self.spoken_reply += current_chunk_spoken + " "
                            play_obj.stop()
                            break

                        time.sleep(0.1)  # Prevent busy-waiting # frequency of visualizer

                    # If no interruption occurred during this chunk playback
                    if not self.interruption:
                        self.spoken_reply += chunk + " "

                print("\nSpoken text:", self.spoken_reply)
                self.remaining_reply = f"(Voice reply interrupted, remaining unsaid reply)\n{self.remaining_reply}"
                print("\nRemaining text:", self.remaining_reply)

                return self.spoken_reply, self.remaining_reply
            except Exception as e:
                print(f"Error in text-to-speech: {e}")
        

    def monitor_audio_interruption(self):
        """
        Continuously monitors for user interruptions via keyboard input.
        Runs in a separate thread to enable real-time interruption detection.
        """

        while True:
            if keyboard.is_pressed("space"):
                self.interruption = True
            else:
                self.interruption = False
            time.sleep(0.1)  # Prevent busy-waiting


    def exit_gracefully(self):
        # TODO: Implement graceful exit logic
        pass
    
    
    
    async def process_and_create_chat_generation(
        self,
        message: MessageSegment,
        is_api_request: bool = False,
        is_audio_requested_in_api_response: bool = False,
        is_audio_in_api_request: bool = False,
        is_vision_enabled: bool = False,
        if_camera_feed: bool = False,
        if_video_file: bool = False,
        if_screenshot_or_image: bool = False,
        image_urls: list[str] | None = None,
        video_urls = list[str] | None= None,
        audio_urls = list[str] | None= None,
        file_urls = list[str] | None = None,
    ) -> str:
        """
        Process the user's message, create a chat segment, and call for reply generation.
        This method handles the conversation history, generates a response, and manages voice replies if enabled.
        It also supports vision-enabled features and different types of media inputs.
        The method is designed to be called asynchronously.
        It also handles api request and audio request in the response.

        Args:
            message (str): takes single message segment
            is_api_request (bool, optional): _description_. Defaults to False.
            is_vision_enabled (bool, optional): _description_. Defaults to False.
            if_camera_feed (bool, optional): _description_. Defaults to False.
            if_video_file (bool, optional): _description_. Defaults to False.
            if_screenshot (bool, optional): _description_. Defaults to False.
            image_urls list[str]: List of image urls. Defaults to [].

        Raises:
            e: _description_

        Returns:
            dict: Reply from the assistant {message: str or {text: str, audio: audio}, timestamp: str}
        """
        
        
        if not is_vision_enabled:
            if_camera_feed = False
            if_video_file = False
            if_screenshot_or_image = False
            image_urls = None
            if is_audio_in_api_request:
                text = ""
                # transcribe audio to text from audio_url
            else:
                text = message.content
            
            user_message = [
                {
                    "role": message.role,
                    "content": text,
                    type: "user_message",
                    "timestamp": message.timestamp,
                }
            ]
        
        if is_vision_enabled:
            message_content = []
            if if_camera_feed:
                message_content.append({"type": "text", "text": "These are multiple frames a video segment from your camera."})
                _type = "user_message_with_camera_feed"
            if if_video_file:
                message_content.append({"type": "text", "text": "These are multiple frames of an uploaded video file."})
                _type = "user_message_with_video_file"
            
            if if_video_file or if_camera_feed:
                frame_urls = [] # TODO: call process_video from video_urls
                # Add image URLs to the message content
                for frame_url in frame_urls:
                    message_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"{frame_url}"
                                },
                            })
                    
            # Add text message to address the images in message content
            _type: str = "user_message"
            if if_screenshot_or_image:
                message_content.append({"type": "text", "text": "This is a screenshot."})
                _type = "user_message_with_screenshot"
                # Add image URLs to the message content
                for image_url in image_urls:
                    message_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"{image_url}"
                                },
                            })
                    
        
            # Append the user's text message with the image message content
            if message.content is type(str):
                message_content.append({"type": "text", "text": message.content})

            user_message = [
                {
                    "role": "user",
                    "content": message_content,
                    type: _type,
                    "timestamp": message.timestamp,
                }
            ]
                
        
        recent_conversation = []
        if self.conversation_history_engine:
            try:
                recent_conversation = (
                    await self.conversation_history_engine.get_recent_conversation()
                )
                recent_conversation = recent_conversation + user_message
                asyncio.run(self.conversation_history_engine.add_conversation(user_message))
            except Exception as e:
                print(f"Error getting recent conversation: {e}")
                raise e
        else:
            recent_conversation = recent_conversation + user_message

        response = await self.inference_processor.create_chat_completion(
            recent_conversation
        )
        
        # TODO: if gui enabled - callback print or stream in ui

        if self.conversation_history_engine:
                asyncio.run(self.conversation_history_engine.add_conversation(
                    [
                        {
                            "role": "assistant",
                            "content": response,
                            type: "computer_message",
                            "timestamp": datetime.now().isoformat(),
                        }
                    ]
                ))
        if not self.voice_reply_enabled or is_api_request:
            return MessageSegment(role="assistant", message=response, timestamp=datetime.now().isoformat())
            
        # voice reply
        if is_api_request and is_audio_requested_in_api_response:
            audio_reply = self.voice_reply(response, is_audio_requested_in_api_response)
            response = [
                {"type": "audio", "content": audio_reply},
                {"type": "text", "content": response},
            ]
            return MessageSegment(role="assistant", message=response, timestamp=datetime.now().isoformat())

            
        if self.voice_reply_enabled:
            # Speak the Computer's reply with interruption handling
            spoken_reply, unspoken_reply = self.voice_reply(response, is_audio_requested_in_api_response)
            if self.conversation_history_engine:
                self.conversation_history_engine.add_conversation(
                    [
                        {
                            "role": "assistant",
                            "content": spoken_reply,
                            type: "spoken_computer_message",
                            "timestamp": datetime.now().isoformat(),
                        }
                    ]
                )
                    
                if self.remaining_reply != "":
                    self.conversation_history_engine.add_conversation(
                        [
                            {
                                "role": "assistant",
                                "content": unspoken_reply,
                                type: "unspoken_remaining_computer_message",
                                "timestamp": datetime.now().isoformat(),
                            }
                        ]
                    )

            # Return the spoken reply and unspoken reply
            return MessageSegment(
                role="assistant", 
                message=f"{spoken_reply}\n[- Unspoken Remaining Reply: {unspoken_reply} -]",
                timestamp=datetime.now().isoformat()
            )

    async def _run_withput_gui(self, is_test: bool = False):
        """
        Main execution loop for the assistant, without gui.

        Args:
            is_test (bool): If True, runs in test mode with sample response
        """

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
        """Run the assistant in an async event loop"""
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

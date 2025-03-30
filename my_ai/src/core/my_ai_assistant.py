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

from typing import Any, List, Tuple, Optional

from src.inference_engine.inference_processor import InferenceProcessor
from src.memory_processor.conversation_history_engine import ConversationHistoryEngine
from src.speech_engine.speech_engine import SpeechEngine
from my_ai.src.utils.my_ai_utils import load_config
from src.core.api_server.data_models import ContentSegment, MessageContent


class MyAIAssistant:

    def __init__(
        self,
        inference_processor: InferenceProcessor,
        speech_engine: SpeechEngine = None,
        conversation_history_engine: ConversationHistoryEngine = None,
        mock_conversation_history: List[MessageContent] = None
    ):
        self.computer = None
        self.config = load_config()
        self.inference_processor = inference_processor
        self.conversation_history_engine = conversation_history_engine
        self.if_gui_enabled = self.config.get("gui_enabled", False)
        self.voice_reply_enabled = self.config.get("voice_reply_enabled", False)

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
        else:
            print("Speech engine is not initialized.")
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
                    # Generate audio samples using TTS engine
                    samples, sample_rate = self.tts_engine.engine.create(
                        [text], voice="am_adam", speed=0.92, lang="en-us"
                    )

                    # Create in-memory file-like object
                    audio_buffer = io.BytesIO()

                    # Save as WAV file in memory
                    with wave.open(audio_buffer, "wb") as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 2 bytes per sample
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(samples.tobytes())

                    # Get the audio data as bytes
                    audio_data = audio_buffer.getvalue()
                    audio_buffer.close()

                    return audio_data

                except Exception as e:
                    print(f"Error generating audio for API response: {e}")
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
                            print("Interruption detected!")
                            self.remaining_reply = " ".join(chunks[i:])
                            break

                        samples, self.kokoro_sample_rate = (
                            self.tts_engine.engine.create(
                                [chunk], voice="am_adam", speed=0.92, lang="en-us"
                            )
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

                            time.sleep(
                                0.1
                            )  # Prevent busy-waiting # frequency of visualizer

                        # If no interruption occurred during this chunk playback
                        if not self.interruption:
                            self.spoken_reply += chunk + " "

                    print("\nSpoken text:", self.spoken_reply)
                    self.remaining_reply = f"(Voice reply interrupted, remaining unsaid reply)\n{self.remaining_reply}"
                    print("\nRemaining text:", self.remaining_reply)

                    return self.spoken_reply, self.remaining_reply
                except Exception as e:
                    print(f"Error in text-to-speech: {e}")
        else:
            print("Speech engine is not initialized.")
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

    def exit_gracefully(self):
        # TODO: Implement graceful exit logic
        pass

    async def process_and_create_chat_generation(
        self,
        message: MessageContent,
        is_api_request: bool = False,
        is_audio_requested_in_api_response: bool = False,
        is_vision_enabled: bool = False,
        if_camera_feed: bool = False,
    ) -> MessageContent:
        """
        Process the message and create a chat generation.
        Args:
            message (MessageContent): The message content to process
            is_api_request (bool, optional): Flag for API request. Defaults to False.
            is_audio_requested_in_api_response (bool, optional): Flag for audio in API response. Defaults to False.
            is_vision_enabled (bool, optional): Flag for vision model. Defaults to False.
            if_camera_feed (bool, optional): Flag for camera feed. Defaults to False.
        Returns:
            MessageContent: The processed message content
        """
        response: MessageContent = None
        if if_camera_feed:
            is_vision_enabled = True

        message.type = "user_message"
        
        recent_conversation = self.mock_conversation_history if not self.conversation_history_engine else []
        
        if self.conversation_history_engine:
            try:
                recent_conversation = (
                    await self.conversation_history_engine.get_recent_conversation()
                )
                if len(recent_conversation) > 0:
                    recent_conversation = recent_conversation + message

                await self.conversation_history_engine.add_conversation([message])
            except Exception as e:
                print(f"Error getting recent conversation: {e}")
                raise e
        else:
            recent_conversation = message

        _response = await self.inference_processor.create_chat_completion(
            messages=recent_conversation,
            if_vision_inference=is_vision_enabled,
            if_camera_feed=if_camera_feed,
        )

        response = MessageContent(
            role="assistant",
            message=_response,
            timestamp=datetime.now().isoformat(),
            type="computer_response",
        )

        if self.conversation_history_engine:
            await self.conversation_history_engine.add_conversation([response])

        # respond
        if is_api_request:
            if not is_audio_requested_in_api_response:
                return response

            # audio in api response
            else:
                if self.speech_engine != None:
                    audio_reply = self.voice_reply(
                        response, is_audio_requested_in_api_response
                    )
                    response = [
                        ContentSegment(
                            type="text",
                            content=_response,
                            description="text reply",
                        ),
                        ContentSegment(
                            type="text",
                            audio_url=audio_reply,
                            description="audio of the text reply",
                        ),  # TODO: fix it properly according to ContentSegment class
                    ]
                    return MessageContent(
                        role="assistant",
                        message=response,
                        timestamp=datetime.now().isoformat(),
                        type="computer_response",
                    )

        if not is_api_request and self.voice_reply_enabled:
            if self.speech_engine != None:
                # Speak the Computer's reply with interruption handling
                spoken_reply, unspoken_reply = self.voice_reply(
                    _response,
                    is_audio_requested_in_api_response,  # TODO: only text from _response
                )
                if self.remaining_reply != "":
                    reply = f"{spoken_reply}"
                else:
                    reply = f"{spoken_reply}\n[- Interrupted, Remaining Unspoken Reply: {unspoken_reply} -]"

                if self.conversation_history_engine:
                    self.conversation_history_engine.add_conversation(
                        [
                            MessageContent(
                                role="assistant",
                                message=ContentSegment(type="text", content=reply),
                                timestamp=datetime.now().isoformat(),
                                type="computer_message",
                            ),
                        ]
                    )

        # Return the spoken reply and unspoken reply, for gui or direct call from other process
        return MessageContent(
            role="assistant",
            message=f"{spoken_reply}\n[- Unspoken Remaining Reply: {unspoken_reply} -]",
            timestamp=datetime.now().isoformat(),
        )

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

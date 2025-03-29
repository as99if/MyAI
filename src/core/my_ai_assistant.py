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
# import gradio as gr
from typing import Tuple, Optional
# from src.core.visualizer import SpectrogramWidget
# from src.core.visualizer_ import generate_dotted_spectrogram
from src.utils.utils import load_config


class MyAIAssistant:
    
    def __init__(self, inference_processor, speech_engine, vision_history_engine = None, conversation_history_engine = None):
        self.computer = None
        self.config = load_config()
        self.inference_processor = inference_processor
        self.conversation_history_engine = conversation_history_engine
        self.if_gui_enabled = self.config.get('gui_enabled', False)
        self.voice_reply_enabled = self.config.get('voice_reply_enabled', False)
        
        self.tts_engine = speech_engine.tts_engine
        self.asr_engine = speech_engine.asr_engine

        # PyAudio configuration for audio detection
        self.CHUNK = 2048 # 1024, 4096
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1 # mono
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
            daemon=True
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
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

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
            while keyboard.is_pressed('space'):
                if stream is None:
                    stream = self.pyaudio.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK,
                        stream_callback=audio_callback
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
        if len(frames) > 0 :
            # Append silent frames
            # for _ in range(num_silent_frames):
            #    frames.append(silent_frame)

            # Save the recorded audio to a temporary WAV file
            with wave.open(self.WAVE_OUTPUT_FILENAME, 'wb') as wf:
                print("Processing audio...")
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))

            try:
                # Use Whisper CPP for transcription
                
                result = self.asr_engine.engine.transcribe(media=self.WAVE_OUTPUT_FILENAME)

                # Clean up temporary file
                if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                    os.remove(self.WAVE_OUTPUT_FILENAME)

                return result[0].text if result else None

            except Exception as e:
                print(f"Transcription error: {e}")
                if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                    os.remove(self.WAVE_OUTPUT_FILENAME)
                return None

    def voice_reply(self, text) -> Tuple[str, str]:
        """
        Converts text to speech with interruption handling.

        Args:
            text (str): Text to be spoken

        Returns:
            Tuple[str, str]: (spoken_reply, remaining_reply)
            - spoken_reply: The portion of text that was successfully spoken
            - remaining_reply: Unspoken text if interrupted
        """
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
                    self.remaining_reply = ' '.join(chunks[i:])
                    break

                samples, self.kokoro_sample_rate = self.tts_engine.engine.create(
                    [chunk], voice="am_adam", speed=0.92, lang="en-us"
                )
                print("Playing audio...")
                wave_obj = simpleaudio.WaveObject(
                    samples, num_channels=1, bytes_per_sample=2, sample_rate=self.kokoro_sample_rate)

                # Play the generated audio and monitor interruptions during playback
                play_obj = wave_obj.play()
                start_time = time.time()

                while play_obj.is_playing():
                    #if True: # if gui enabled
                    #    self.gui_visualizer.update_spectrogram(wave_obj, self.kokoro_sample_rate)
                    if self.interruption:  # Stop playback if user interrupted
                        elapsed_time = time.time() - start_time
                        spoken_chars = int(
                            elapsed_time * self.CHARS_PER_SECOND)
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
            if keyboard.is_pressed('space'):
                self.interruption = True
            else:
                self.interruption = False
            time.sleep(0.1)  # Prevent busy-waiting

    def backup_conversation_history(self):
        """Backup conversation history to a file"""

    def __del__(self):
        """Cleanup method for graceful exit"""
        try:
            # Delete all PyTorch tensors and models
            # for obj in gc.get_objects():
            #    if torch.is_tensor(obj):
            #        del obj
            pass
            

            # Force GPU memory cleanup
            # if torch.cuda.is_available():
            #    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Ensure system resources are released
            gc.collect()

    def exit_gracefully(self):
        """
        Performs cleanup operations before program termination:
        - Stops audio monitoring
        - Releases PyAudio resources
        - Cleans up memory
        - Backs up conversation history
        """

        print("Exiting and clearing loaded model...")
        self.monitor_audio_interruption_thread.join(timeout=1)
        self.pyaudio.terminate()
        # Clear conversation history
        if hasattr(self, 'conversation_history_engine'):
            self.conversation_history_engine.clear()

        # Release TTS and ASR engines
        if hasattr(self, 'tts_engine'):
            del self.tts_engine
        if hasattr(self, 'asr_engine'):
            del self.asr_engine

        # Clear audio resources
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()

            # Remove temporary files
        if os.path.exists(self.WAVE_OUTPUT_FILENAME):
            os.remove(self.WAVE_OUTPUT_FILENAME)
        
        # delete pytorch memory pointers
        # self.__del__()
        # backup conversation memory
        exit(0)

    async def process_and_create_chat_generation(self, message: str, is_api_request: bool = False, is_audio_reply_api_request: bool = False):
        """
        Process a user message and generate an AI chat response with optional voice reply.

        This method handles:
        1. Creating a formatted user message with timestamp
        2. Retrieving recent conversation history
        3. Generating AI response
        4. Managing voice replies if enabled
        5. Storing conversation history

        Args:
            message (str): The user's input message to process
            is_api_request (bool, optional): Flag to indicate if request is from API. 
                If True, skips voice reply. Defaults to False.

        Returns:
            str: The AI generated response text

        Raises:
            Exception: If there's an error retrieving conversation history

        Example:
            ```python
            response = await assistant.process_and_create_chat_generation("Hello!", is_api_request=False)
            print(response)  # Prints AI response
            ```
        """
        user_message =[{"role": "user", "content": message, type: "user_message", "timestamp": datetime.now().isoformat()}]
            # if vlm
            # {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "image_url",
            #             "image_url": {
            #                 "url": f"{image_url}"
            #             },
            #         },
            #         {"type": "text", "text": message},
            #     ],
            # }
        recent_conversation = []
        if self.conversation_history_engine:
            try: 
                recent_conversation = await self.conversation_history_engine.get_recent_conversation()
                recent_conversation = recent_conversation + user_message
                self.conversation_history_engine.add_conversation(user_message)
            except Exception as e:
                print(f"Error getting recent conversation: {e}")
                raise e
        else:
            recent_conversation = recent_conversation + user_message
            # print of stream in gradio ui
            # if self.if_gui_enabled:
            #     self.update_chat_interface(message, is_user=True)

        response = await self.inference_processor.create_chat_completion(recent_conversation)
        
        # TODO later: for image frames in message body

            # voice reply
        if self.voice_reply_enabled:
            if not is_api_request or is_audio_reply_api_request:
                # Speak the Computer's reply with interruption handling
                spoken_reply, unspoken_reply = self.voice_reply(response)
                if self.conversation_history_engine:
                    self.conversation_history_engine.add_conversation([
                        {
                            "role": "assistant", "content": spoken_reply, type: "spoken_computer_message", "timestamp": datetime.now().isoformat()                        }
                        ])
                    if self.remaining_reply != "":
                        self.conversation_history_engine.add_conversation([
                            {
                                "role": "assistant", "content": unspoken_reply, type: "unspoken_remaining_computer_message", "timestamp": datetime.now().isoformat()
                            }
                        ])
                print(f"Computer Reply: {spoken_reply}")
                    # print of stream in gradio ui
                    # if self.if_gui_enabled:
                    #     self.update_chat_interface(spoken_reply, is_user=False)
                    #     self.update_chat_interface(unspoken_reply, is_user=False)
        else:
            print(f"Computer Reply: {response}")
            if self.conversation_history_engine:
                self.conversation_history_engine.add_conversation([
                    {
                        "role": "assistant", "content": response, type: "computer_message", "timestamp": datetime.now().isoformat()
                    }
                ])
            # print of stream in gradio ui
            # if self.if_gui_enabled:
            #     self.update_chat_interface(response, is_user=False)
        return response
    
    async def __run__(self, is_test: bool = False):
        """
        Main execution loop for the assistant.

        Args:
            is_test (bool): If True, runs in test mode with sample response
        """
        
        # memory procesor and conversation summarizer and all those things will run on call of this function
        # maybe for creating conversation summary every  weeks, we can use a timer to call the function that creates the summary to make context smaller
        
        if is_test:
            self.test_voice_reply()
            return
            
        print("Listening... (Hold space to record)")
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
        """
        self.computer = threading.Thread(target=self.__run__, name="computer")
        self.computer.start()
        self.computer.join()
        """
        
        """Run the assistant in an async event loop"""
        try:
            # Create new event loop for the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the coroutine
            loop.run_until_complete(self.__run__())
        except Exception as e:
            print(f"Error in run: {e}")
        finally:
            loop.close()


    def test_voice_reply(self):
        text = "This is a test message, or text. Whatever.."
        self.voice_reply(text=text)
        


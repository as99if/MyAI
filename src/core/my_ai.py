# TODO: conversation history
# stt bad
# tts noisy

import os
from datetime import datetime
import whisper
import pyaudio
import wave
import numpy as np
import queue
import time
from pydub import AudioSegment
import math
import simpleaudio as sa
import io
import keyboard  # Add this import
import gc
import threading
import time
import signal
import torch
from whisper_cpp_python import Whisper
import pprint
from omegaconf import OmegaConf
from src.core.clear_memory import clear_memory
from src.memory.conversation_history_service import ConversationHistoryEngine
import asyncio
import sounddevice as sd
from kokoro_onnx import Kokoro


class MyAIAssistant:
    def __init__(self, config, inference_engine, conversation_history_service):
        self.config = config
        self.inference_engine = inference_engine
        self.conversation_history_service = conversation_history_service
        self.voice_reply_enabled = self.config.get(
            'voice_reply_enabled', False)
        self.whisper = None

        self.init_speech_models()

        # PyAudio configuration for audio detection
        self.CHUNK = 1024 * 4
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # 44100
        self.WAVE_OUTPUT_FILENAME = "temp_audio.wav"
        self.p = pyaudio.PyAudio()

        # Audio detection parameters
        self.THRESHOLD = 500
        self.interruption = False

        # Text tracking for interruption handling
        self.spoken_text = ""
        self.remaining_text = ""

        # Average speaking rate (characters per second)
        self.CHARS_PER_SECOND = 15

        # Start the interruption monitoring thread
        self.monitor_interruption_thread = threading.Thread(
            target=self.monitor_interruption,
            name="monitor_interruption",
            daemon=True
        )
        # Interrupting monitoring
        self.monitor_interruption_thread.start()

        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        self.run()

    def init_speech_models(self):
        # Load Whisper model for speech-to-text
        self.whisper = Whisper(model_path="./models/ggml-tiny.bin")
        self.params.n_threads = 2
        self.params.print_special = False
        self.params.print_progress = True
        self.params.print_realtime = True
        self.params.print_timestamps = False

    def split_text_into_chunks(self, text, chunk_size=100):
         """Split text into chunks for progressive TTS"""
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

    def listen(self, duration=5):
        """Record and transcribe user speech"""
        audio_queue = queue.Queue()
        frames = []

        def audio_callback(in_data, frame_count, time_info, status):
            audio_queue.put(in_data)
            return (None, pyaudio.paContinue)

        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=audio_callback
        )

        print("Listening... (Hold space to record)")
        while keyboard.is_pressed('space'):
            if not audio_queue.empty():
                data = audio_queue.get()
                frames.append(data)

        print("Processing audio...")
        stream.stop_stream()
        stream.close()

        # Save the recorded audio to a temporary WAV file
        with wave.open(self.WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))

        try:
            # Use Whisper CPP for transcription
            result = self.whisper.transcribe(self.WAVE_OUTPUT_FILENAME)

            # Clean up temporary file
            if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                os.remove(self.WAVE_OUTPUT_FILENAME)

            return result.text if result else None

        except Exception as e:
            print(f"Transcription error: {e}")
            if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                os.remove(self.WAVE_OUTPUT_FILENAME)
            return None

    def voice_reply(self, text):
        """Speak text using Silero TTS with interruption handling"""
        try:
            # Reset flags and text tracking
            self.interruption = False
            self.spoken_text = ""
            self.remaining_text = text

            # Split text into manageable chunks
            chunks = self.split_text_into_chunks(text)

            # detect kez press for interruption handling

            # Process each chunk of text for TTS synthesis and playback
            for i, chunk in enumerate(chunks):
                if self.interruption:  # Check if user interrupted by speaking
                    print("Interruption detected!")
                    self.remaining_text = ' '.join(chunks[i:])
                    break

                # Generate audio using Silero TTS
                kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
                samples, self.kokoro_sample_rate = kokoro.create(
                    [chunk], voice="af_sarah", speed=1.0, lang="en-us"
                )
                print("Playing audio...")
                wave_obj = sa.WaveObject(
                    samples, num_channels=1, bytes_per_sample=2, sample_rate=self.kokoro_sample_rate)

                # Play the generated audio and monitor interruptions during playback
                play_obj = wave_obj.play()
                start_time = time.time()

                while play_obj.is_playing():
                    if self.interruption:  # Stop playback if user speaks
                        elapsed_time = time.time() - start_time
                        spoken_chars = int(
                            elapsed_time * self.CHARS_PER_SECOND)
                        current_chunk_spoken = chunk[:spoken_chars]
                        self.spoken_text += current_chunk_spoken + " "
                        play_obj.stop()
                        break

                    time.sleep(0.1)  # Prevent busy-waiting

                # If no interruption occurred during this chunk playback
                if not self.interruption:
                    self.spoken_text += chunk + " "

            print("\nSpoken text:", self.spoken_text)
            self.remaining_text = f"(Voice reply interrupted, remaining unsaid reply)\n{self.remaining_text}"
            print("\nRemaining text:", self.remaining_text)
            if self.conversation_history_service:
                self.conversation_history_service.add_conversation([
                    ('assistant', self.spoken_text)
                ])
                if self.remaining_text != "":
                    self.conversation_history_service.add_conversation([
                    ('assistant', self.remaining_text)
                ])
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def monitor_interruption(self):
        """Continuously check if space key is pressed"""
        while True:
            if keyboard.is_pressed('space'):
                self.interruption = True
            else:
                self.interruption = False
            time.sleep(0.1)  # Prevent busy-waiting

    def exit_gracefully(self, signum, frame):
        """Handle exiting the program gracefully"""
        print("Exiting and clearing loaded model...")
        self.monitor_interruption_thread.join(timeout=1)
        self.p.terminate()
        clear_memory()
        # backup conversation memory
        self.inference_engine.conversation_history_service.backup_conversation_history()
        exit(0)

    def run(self):

        while True:
            message = self.listen()
            if message is None:
                continue

            print("Transcription:", message)
            # TODO: correction with llm
            prompt = f"This is a message or a command from the user for you as an voice AI assistant:\n '{message}'\n There could be mistakes due to voice recognition or audio detection. Correct it."

            if not self.voice_reply_enabled:
                response = self.inference_engine.chat_completion(prompt)
                if self.conversation_history_service:
                    self.conversation_history_service.save_chat_segment([
                        ('human', message),
                        ('assistant', response)
                    ])
            else:
                # response = self.inference_engine.chat_completion(message)
                # print(f"AI Reply: {response}")

                # Speak the AI's reply with interruption handling
                # self.voice_reply(response)
                print("AI Corrected Transcription:", response)

            if "exit" in message.lower():
                print("Exiting...")
                clear_memory()
                break
            

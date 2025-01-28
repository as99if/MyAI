# TODO: conversation history
# stt bad
# tts noisy

import os
from datetime import datetime
import pyaudio
import wave
import numpy as np
import queue
import time
import gc
import torch
from pydub import AudioSegment
import math
import simpleaudio as sa
import io
import keyboard  # Add this import
import gc
import threading
import time
import signal
import pprint


class MyAIAssistant:
    def __init__(self, config, inference_engine, conversation_history_engine, asr_engine, tts_engine, vision_history_engine: None):
        self.computer = None
        self.config = config
        self.inference_engine = inference_engine
        self.conversation_history_engine = conversation_history_engine
        self.voice_reply_enabled = self.config.get(
            'voice_reply_enabled', False)

        self.tts_engine = tts_engine
        self.asr_engine = asr_engine

        # PyAudio configuration for audio detection
        self.CHUNK = 1024 * 4
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # 44100
        self.WAVE_OUTPUT_FILENAME = "temp_audio.wav"
        self.pyaudio = pyaudio.PyAudio()

        # Audio detection parameters
        self.THRESHOLD = 500
        self.interruption = False

        # Text tracking for interruption handling
        self.spoken_text = ""
        self.remaining_text = ""

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

        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

        self.run()

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

        stream = self.pyaudio.open(
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
            wf.setsampwidth(self.pyaudio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))

        try:
            # Use Whisper CPP for transcription
            result = self.asr_engine.engine.transcribe(
                self.WAVE_OUTPUT_FILENAME, new_segment_callback=print)

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

                samples, self.kokoro_sample_rate = self.tts_engine.engine.create(
                    [chunk], voice="am_adam", speed=0.92, lang="en-us"
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
            
            return self.spoken_text, self.remaining_text
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def monitor_camera_loop(self):
        """Monitor the camera loop for interruptions"""
        while True:
            if keyboard.is_pressed('.'):
                self.interruption = True
            else:
                self.interruption = False
            time.sleep(0.1)
    
    
    def monitor_audio_interruption(self):
        """Continuously check if space key is pressed"""
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
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    del obj

            # Clear conversation history
            if hasattr(self, 'conversation_history_service'):
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

            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Ensure system resources are released
            gc.collect()

    def exit_gracefully(self):
        """Handle exiting the program gracefully"""
        
        print("Exiting and clearing loaded model...")
        self.monitor_audio_interruption_thread.join(timeout=1)
        self.pyaudio.terminate()
        self.__del__()
        # backup conversation memory
        exit(0)

    def __run__(self):
        # memory procesor and conversation summarizer and all those things will run on call of this function
        # maybe for creating conversation summary every  weeks, we can use a timer to call the function that creates the summary to make context smaller
        # self.memory_processor.start_memory_processing()
        while True:
            message = self.listen()
            if message is None:
                continue

            print("Transcription:", message)
            if self.conversation_history_engine:
                self.conversation_history_engine.add_conversation([
                    {
                        "role": "human", "content": message, type: "user_message", "timestamp": datetime.now().isoformat()
                    },
                ])
            response = self.inference_engine.chat_completion(message)
            # other data to keep in history
            response = response['choices'][0]['message']['content']
            if self.voice_reply_enabled:
                # Speak the Computer's reply with interruption handling
                spoken_reply, unspoken_reply = self.voice_reply(response)
                if self.conversation_history_engine:
                    self.conversation_history_engine.add_conversation([
                        {
                            "role": "computer", "content": self.spoken_text, type: "spoken_computer_message", "timestamp": datetime.now().isoformat()
                        }
                    ])
                    if self.remaining_text != "":
                        self.conversation_history_engine.add_conversation([
                            {
                            "role": "computer", "content": self.spoken_text, type: "unspoken_remaining_computer_message", "timestamp": datetime.now().isoformat()
                        }
                    ])
                print(f"Computer Reply: {spoken_reply}")
                
            else:
                print(f"Computer Reply: {response}")
                if self.conversation_history_engine:
                    self.conversation_history_engine.add_conversation([
                        {
                            "role": "computer", "content": response, type: "computer_message", "timestamp": datetime.now().isoformat()
                        }
                    ])
                
            
                
            if "clean and shutdown" in message.lower():
                print("Exiting...")
                self.exit_gracefully()
                break
    
    def run(self):
        self.computer = threading.Thread(target=self.__run__, name="computer")
        self.computer.start()
        self.computer.join()
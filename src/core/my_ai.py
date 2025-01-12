# TODO: conversation history 
# stt bad
# tts noisy

import os
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
from silero import silero_stt, silero_tts, silero_te
import pprint
from omegaconf import OmegaConf
from src.core.clear_memory import clear_memory
from src.memory.conversation_history_service import ConversationHistoryEngine

class MyAIAssistant:
    def __init__(self, config, inference_engine, conversation_history_service):
        self.config = config
        self.inference_engine = inference_engine
        self.conversation_history_service = conversation_history_service
        self.voice_reply_enabled = self.config.get('voice_reply_enabled', False)
        self.device = torch.device('cpu')
        # self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        # pytorch 2.5 and whisper issue with mps
        print(f"Using device: {self.device}")
        # self.init_speech_models()
        
        # PyAudio configuration for audio detection
        self.CHUNK = 1024 * 4
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000 # 44100
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
        # self.monitor_interruption_thread.start()
        
        
        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        
        self.run()
    
    def init_speech_models(self):
        # Load Whisper model for speech-to-text
        # print("Loading Whisper model...")
        # self.whisper_model = whisper.load_model("base", device=self.device)
        # Load Silero ASR model for speech-to-text
        print("Loading Silero ASR model...")
        # https://github.com/snakers4/silero-models/blob/master/models.yml
        self.silero_asr_model, self.silero_decoder, self.silero_utils = silero_stt(
            jit_model='jit_xlarge', # jit, onnx, jit_q, onnx_xlarge
            language='en',
            device=self.device,
        )
        
        # Load Silero TTS model for text-to-speech
        print("Loading Silero TTS model...")
        # https://github.com/snakers4/silero-models/blob/master/models.yml
        # if speaker == 'lj_16khz'
        # self.silero_tts_model, self.silero_symbols, self.silero_sample_rate, _, self.apply_silero_tts = silero_tts(
        self.silero_tts_model, _ = silero_tts(
             language='en',
             speaker= 'v3_en', # lj_16khz'
             device=self.device
        )
        # use torch to load silero models from local dir: https://colab.research.google.com/github/snakers4/silero-models/blob/master/examples.ipynb#scrollTo=jc7ZqfooYZnD
        print("Silero model loaded successfully")
    
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
        
        
        while keyboard.is_pressed('space'):
            if not audio_queue.empty():
                for _ in range(0, int(self.RATE / self.CHUNK * duration)):
                    data = audio_queue.get()
                    frames.append(data)
        
            #print("Finished recording")
        
        stream.stop_stream()
        stream.close()
        
        # Combine frames into a single byte buffer
        audio_buffer = b''.join(frames)
    
        # Use an in-memory buffer for the audio data
        audio_segment = AudioSegment(
            data=audio_buffer,
            sample_width=self.p.get_sample_size(self.FORMAT),
            frame_rate=self.RATE,
            channels=self.CHANNELS
        )
        
        # Convert audio segment to the required format
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        audio_samples = np.array(audio_segment.get_array_of_samples())
        audio_tensor = torch.from_numpy(audio_samples).float().unsqueeze(0)  # Add batch dimension
        if len(audio_samples) == 0:
            return None
        
        # Ensure the tensor is on the same device as the model
        audio_tensor = audio_tensor.to(self.device)
        # Use Silero ASR model for transcription
        with torch.no_grad():
            output = self.silero_asr_model(audio_tensor)
            # Use Whisper model for transcription
            # result = self.whisper_model.transcribe(audio_samples.cpu().numpy())
    
        #print(output)
        result = self.silero_decoder(output[0])
        
        return result
    
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
                #audio_tensor = self.apply_silero_tts(texts=[chunk], model=self.silero_tts_model,
                audio_tensor = self.silero_tts_model(texts=[chunk], model=self.silero_tts_model,
                                                sample_rate=24000, # sample_rate: [8000, 24000, 48000] # if speaker == lj_16khz, then self.silero_sample_rate
                                                device=self.device,
                                                symbols=self.silero_symbols,
                                            )[0]
                wave_obj = sa.WaveObject(audio_tensor.cpu().numpy().tobytes(), num_channels=1, bytes_per_sample=2, sample_rate=self.silero_sample_rate)
                
                # Play the generated audio and monitor interruptions during playback
                play_obj = wave_obj.play()
                start_time = time.time()
                   
                while play_obj.is_playing():
                    if self.interruption:  # Stop playback if user speaks
                        elapsed_time = time.time() - start_time
                        spoken_chars = int(elapsed_time * self.CHARS_PER_SECOND)
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
                if self.remaining_text is not "":
                    self.conversation_history_service.add_conversation([
                    ('assistant', self.remaining_text)
                ])
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def monitor_interruption(self):
        """Continuously check if space key is pressed"""
        # TODO: put silero VAD https://github.com/snakers4/silero-vad instead of keypressing
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
        # temporarilty clear up conversation memory
        cm = ConversationHistoryEngine(self.config)

        cm.get_all_conversations()
        print(cm)
        
        message = "write counting of one to ten, and then explain what is furier mathematics in three sentences. Do no write more than seven sentences."
        response_text = self.inference_engine.chat_completion(message)
        print(response_text)
        if self.conversation_history_service:
                self.conversation_history_service.add_conversation([
                    ("human", message),
                    ("assistant", response_text)
                ])
        
        
        """while True:
            message = self.listen()
            if message is None:
                continue
            
            print("Transcription:", message)
            # TODO: correction with llm
            prompt = f"This is a message or a command from the user for you as an voice AI assistant:\n '{message}'\n There could be mistakes due to voice recognition or audio detection. Correct it."
            
            if !self.voice_reply_enabled:
                response = self.inference_engine.chat_completion(prompt)
                if self.conversation_history_service:
                    self.conversation_history_service.save_chat_segment([
                        ('human', message),
                        ('assistant', response)
                    ])
            else:
                #response = self.inference_engine.chat_completion(message)
                #print(f"AI Reply: {response}")
            
                # Speak the AI's reply with interruption handling
                # self.voice_reply(response)
                print("AI Corrected Transcription:", response)
            
            if "exit" in message.lower():
                print("Exiting...")
                clear_memory()
                break
            
            
            )"""
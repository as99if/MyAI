"""
@author - Aisf Ahmed - asif.shuvo2199@outlook.com
"""
import os

import io
import queue
from typing import Optional
import wave
import keyboard
import numpy as np
import threading
import time
from scipy.fft import fft
from scipy.signal import get_window
from piper import PiperVoice, SynthesisConfig
import pygame

import pyaudio
import wave


class TTSEngine:
    def __init__(self, model_path="/home/asifahmedshuvo/Development/MyAI/my_ai_assistant/models/speech/en_GB-semaine-medium.onnx", visualizer=None):
        self.voice = PiperVoice.load(model_path)
        self.syn_config = SynthesisConfig(
            volume=0.5,
            length_scale=1.0,
            noise_scale=1.0,
            noise_w_scale=1.0,
            normalize_audio=False,
        )
        # Visualization parameters
        self.visualizer = visualizer  # PyQt AudioVisualizer
        self.num_bands = 32
        self.led_levels = 32
        self.user_interrupted = [False]
        # Start the interruption monitoring thread
        self.monitor_audio_interruption_thread = threading.Thread(
            target=self.monitor_audio_interruption,
            name="monitor_audio_interruption",
            daemon=True,
        )
        # Interrupting monitoring
        self.monitor_audio_interruption_thread.start()

    def set_visualizer(self, visualizer):
        self.visualizer = visualizer

    def get_visualizer(self):
        return self.visualizer
    
    def monitor_audio_interruption(self):
        """
        Continuously monitors for user interruptions via keyboard input.
        Runs in a separate thread to enable real-time interruption detection.
        """
        while True:
            if keyboard.is_pressed("space"):
                self.user_interrupted = [True]
            else:
                    
                self.user_interrupted = [False]
            time.sleep(0.1)  # Prevent busy-waiting
        
    
    def synthesize(self, text):
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file=wav_file, syn_config=self.syn_config)
        audio_buffer.seek(0)
        return audio_buffer

    def play_synthesized_audio_with_led_visualizer(self, text) -> tuple:
        """
        Play synthesized audio with LED visualizer and interruption support.
        Visualization is done via the PyQt AudioVisualizer widget.
        """
        audio_buffer = self.synthesize(text)
        with wave.open(audio_buffer, "rb") as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            total_frames = wav_file.getnframes()

        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.float32

        audio_data = np.frombuffer(frames, dtype=dtype)
        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2)
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)
        if dtype != np.float32:
            audio_data = audio_data / np.max(np.abs(audio_data))
        audio_buffer.seek(0)

        pygame.mixer.init(frequency=sample_rate)

        num_bands = self.num_bands
        led_levels = self.led_levels
        chunk_duration = 0.05
        chunk_size = int(sample_rate * chunk_duration)
        freq_bands = np.logspace(np.log10(20), np.log10(sample_rate//2), num_bands + 1)
        band_centers = (freq_bands[:-1] + freq_bands[1:]) / 2
        smoothing_factor = 0.3
        previous_levels = np.zeros(num_bands)
        result_spoken_text = [""]
        result_unspoken_text = [""]

        is_playing = [True]
        self.user_interrupted = [False]

        def get_current_position():
            if pygame.mixer.music.get_busy():
                return pygame.mixer.music.get_pos() / 1000.0
            return 0

        def calculate_text_split():
            pos_seconds = get_current_position()
            total_duration = total_frames / sample_rate
            proportion_played = min(1.0, max(0.0, pos_seconds / total_duration))
            char_position = int(len(text) * proportion_played)
            result_spoken_text[0] = text[:char_position].strip()
            result_unspoken_text[0] = text[char_position:].strip()
            return proportion_played

        def play_audio():
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            try:
                while pygame.mixer.music.get_busy() and not self.user_interrupted[0]:
                    time.sleep(0.01)
                if self.user_interrupted[0]:
                    print("Audio thread detected interruption")
                    
                else:
                    result_spoken_text[0] = text
                    result_unspoken_text[0] = ""
                    print("Playback completed normally")
            except Exception as e:
                print(f"Error in audio thread: {e}")
            finally:
                is_playing[0] = False

        audio_thread = threading.Thread(target=play_audio)
        audio_thread.daemon = True
        audio_thread.start()

        # Main visualization loop (no pygame display, just update PyQt visualizer)
        running = True
        interrupted = False
        while running:
            # --- Synchronize visualization with audio playback ---
            pos_seconds = get_current_position()
            start_idx = int(pos_seconds * sample_rate)
            end_idx = start_idx + chunk_size

            if is_playing[0] and start_idx < len(audio_data):
                if end_idx > len(audio_data):
                    chunk_data = np.zeros(chunk_size)
                    remaining = len(audio_data) - start_idx
                    chunk_data[:remaining] = audio_data[start_idx:]
                else:
                    chunk_data = audio_data[start_idx:end_idx]
                windowed_data = chunk_data * get_window('hann', len(chunk_data))
                fft_data = np.abs(fft(windowed_data))[:chunk_size//2]
                freqs = np.fft.fftfreq(chunk_size, 1/sample_rate)[:chunk_size//2]
                band_levels = np.zeros(num_bands)
                for i in range(num_bands):
                    band_mask = (freqs >= freq_bands[i]) & (freqs < freq_bands[i+1])
                    if np.any(band_mask):
                        band_levels[i] = np.sqrt(np.mean(fft_data[band_mask]**2))
                if np.max(band_levels) > 0:
                    band_levels = band_levels / np.max(band_levels)
                    band_levels = np.log10(band_levels + 0.01) + 2
                    band_levels = np.clip(band_levels, 0, 1)
                previous_levels = (smoothing_factor * band_levels +
                                   (1 - smoothing_factor) * previous_levels)
            else:
                previous_levels *= 0.8

            # Update PyQt visualizer
            if self.visualizer is not None:
                norm_levels = previous_levels / np.max(previous_levels) if np.max(previous_levels) > 0 else previous_levels
                self.visualizer.update_bars(norm_levels)

            # Check for interruption (optional: add a method to interrupt from UI)
            if not is_playing[0] and not interrupted:
                interrupted = True
            if not is_playing[0]:
                running = False

            time.sleep(0.016)  # ~60 FPS

        print("Visualization finished")
        print(f"Final results - Spoken: {len(result_spoken_text[0])} chars, Unspoken: {len(result_unspoken_text[0])} chars")
        return result_spoken_text[0], result_unspoken_text[0]

    # def play_synthesized_audio_without_led_visualizer(self, text):
    #     audio_buffer = self.synthesize(text)
    #     with wave.open(audio_buffer, "rb") as wav_file:
    #         sample_rate = wav_file.getframerate()
    #         total_frames = wav_file.getnframes()
    #     pygame.mixer.init(frequency=sample_rate)

    #     result_spoken_text = [""]
    #     result_unspoken_text = [""]
    #     is_playing = [True]
    #     self.user_interrupted = [False]

    #     def get_current_position():
    #         if pygame.mixer.music.get_busy():
    #             return pygame.mixer.music.get_pos() / 1000.0
    #         return 0

    #     def calculate_text_split():
    #         pos_seconds = get_current_position()
    #         total_duration = total_frames / sample_rate
    #         proportion_played = min(1.0, max(0.0, pos_seconds / total_duration))
    #         char_position = int(len(text) * proportion_played)
    #         result_spoken_text[0] = text[:char_position].strip()
    #         result_unspoken_text[0] = text[char_position:].strip()
    #         return proportion_played

    #     def play_audio():
    #         pygame.mixer.music.load(audio_buffer)
    #         pygame.mixer.music.play()
    #         try:
    #             while pygame.mixer.music.get_busy() and not self.user_interrupted[0]:
    #                 time.sleep(0.01)
    #             if self.user_interrupted[0]:
    #                 print("Audio thread detected interruption")
    #             else:
    #                 result_spoken_text[0] = text
    #                 result_unspoken_text[0] = ""
    #                 print("Playback completed normally")
    #         except Exception as e:
    #             print(f"Error in audio thread: {e}")
    #         finally:
    #             is_playing[0] = False

    #     audio_thread = threading.Thread(target=play_audio)
    #     audio_thread.daemon = True
    #     audio_thread.start()

    #     running = True
    #     interrupted = False
    #     while running:
    #         time.sleep(0.01)
    #         if not is_playing[0] and not interrupted:
    #             interrupted = True
    #         if not is_playing[0]:
    #             running = False

    #     print("Audio finished")
    #     print(f"Final results - Spoken: {len(result_spoken_text[0])} chars, Unspoken: {len(result_unspoken_text[0])} chars")
    #     return result_spoken_text[0], result_unspoken_text[0]
    def exit_gracefully(self):
        """
        Gracefully exit the TTS engine, stopping any ongoing playback and cleaning up resources.
        """
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        if self.visualizer is not None:
            self.visualizer.clear_bars()
        # Start the interruption monitoring thread
        self.monitor_audio_interruption_thread = threading.Thread(
            target=self.monitor_audio_interruption,
            name="monitor_audio_interruption",
            daemon=True,
        )
        # Interrupting monitoring
        self.monitor_audio_interruption_thread.start()


from pywhispercpp.model import Model as Whisper_ccp
class ASREngine:
    def __init__(self):
        self.engine = Whisper_ccp(
            model="large-v3-turbo",
            models_dir="./models/speech/",
            print_realtime=False,
            print_progress=False
        )
        # Audio config
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pyaudio = pyaudio.PyAudio()
        self.WAVE_OUTPUT_FILENAME = "src/speech_engine/temp_audio.flac"


    def record_and_transcribe(self, duration=5) -> Optional[str]:
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

                result = self.engine.transcribe(
                    media=self.WAVE_OUTPUT_FILENAME
                )

                # Clean up temporary file
                if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                    os.remove(self.WAVE_OUTPUT_FILENAME)

                return result[0].text if result else None

            except Exception as e:
                self.logging_manager.add_message(f"Transcription error: {e}", level='INFO', source='SpeechEngine-ASR')
                if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                    os.remove(self.WAVE_OUTPUT_FILENAME)
                return None
    
    

class SpeechEngine:
    
    def __init__(self):
        self.tts_engine = TTSEngine()
        self.asr_engine = ASREngine()


def test_tts():
    text = "Welcome to the world of speech synthesis! This is a demonstration of text-to-speech with a real-time audio visualizer. You can press the spacebar at any time to interrupt the playback."
    from src.interface.speech_engine import TTSEngine
    from src.interface.speech_visualizer import AudioVisualizer

    # In your PyQt UI class:
    audio_display = AudioVisualizer(width=600, height=200)
    
    tts_engine = TTSEngine(visualizer=audio_display)

    # To play and visualize:
    tts_engine.play_synthesized_audio_with_led_visualizer(text)
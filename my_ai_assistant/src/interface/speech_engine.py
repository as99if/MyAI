"""
@author - Aisf Ahmed - asif.shuvo2199@outlook.com
"""
import os
import io
import queue
from typing import Callable, Optional, Tuple, Generator, Union
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
from src.utils.log_manager import LoggingManager
from pywhispercpp.model import Model as Whisper_ccp

class TTSEngine:
    """Text-to-Speech engine with optional streaming and callback support."""
    
    def __init__(self, model_path: str = "/Users/asifahmed/personal/development/MyAI/my_ai_assistant/models/speech/en_GB-semaine-medium.onnx", debug: bool = False):
        """
        Initialize the TTS engine.
        
        Args:
            model_path (str): Path to the Piper voice model
            debug (bool): Enable debug logging
        """
        self.logging_manager = LoggingManager()
        self.logging_manager.add_message("Initiating - TTSEngine", level="INFO", source="TTSEngine")
        
        self.debug = debug
        self.voice = PiperVoice.load(model_path)
        self.syn_config = SynthesisConfig(
            volume=0.5,
            length_scale=1.0,
            noise_scale=1.0,
            noise_w_scale=1.0,
            normalize_audio=False,
        )
        
        # Playback state
        self.user_interrupted = [False]
        self.is_playing = [False]
        
        # Callbacks
        self.text_stream_callback: Optional[Callable[[str], None]] = None
        self.playback_position_callback: Optional[Callable[[float, bool], None]] = None
        self.audio_data_callback: Optional[Callable[[np.ndarray, int, float], None]] = None
        
        pygame.mixer.init()
        
    def _debug_print(self, message: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(f"[TTS DEBUG] {message}")
    
    def set_text_stream_callback(self, callback: Callable[[str], None]):
        """Set callback for streaming text updates."""
        self.text_stream_callback = callback
    
    def set_playback_position_callback(self, callback: Callable[[float, bool], None]):
        """Set callback for playback position updates."""
        self.playback_position_callback = callback
    
    def set_audio_data_callback(self, callback: Callable[[np.ndarray, int, float], None]):
        """Set callback for providing audio data to visualizer."""
        self.audio_data_callback = callback
        
    def synthesize(self, text: str) -> io.BytesIO:
        """
        Synthesize text to audio.
        
        Args:
            text (str): Text to synthesize
            
        Returns:
            io.BytesIO: Audio buffer containing WAV data
        """
        self._debug_print(f"Synthesizing text: '{text[:50]}...' ({len(text)} chars)")
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file=wav_file, syn_config=self.syn_config)
        audio_buffer.seek(0)
        self._debug_print("Audio synthesis completed")
        return audio_buffer

    def play_synthesized_audio(
        self, 
        text: str, 
        stream: bool = False, 
        mute: bool = False
    ):
        """
        Play synthesized audio with optional streaming.
        """
        self._debug_print(f"[TTS DEBUG] play_synthesized_audio: Start Playing synthesized audio - text: {text}")
        _parameters = ""
        if stream:
            _parameters += "Stream: True; "
        if mute:
            _parameters += "Mute: True; "
        self.logging_manager.add_message(f"Synthesizing reply - {_parameters}", level="INFO", source="TTSEngine")
        
        # Synthesize once
        audio_buffer = self.synthesize(text)
        with wave.open(audio_buffer, "rb") as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            total_frames = wav_file.getnframes()

        # Convert to mono float32 (normalized)
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
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data)) if audio_data.size else 0
        if max_abs > 0:
            audio_data /= max_abs
        total_duration = total_frames / sample_rate if sample_rate else 0.0

        # Callback for visualizer
        if self.audio_data_callback:
            self.audio_data_callback(audio_data, sample_rate, total_duration)
        audio_buffer.seek(0)

        # Ensure mixer matches sample rate
        try:
            init_state = pygame.mixer.get_init()
            if init_state:
                current_freq = init_state[0]
                if current_freq != sample_rate:
                    self._debug_print(f"Reinitializing mixer: {current_freq} -> {sample_rate}")
                    pygame.mixer.quit()
                    pygame.mixer.init(frequency=sample_rate)
            else:
                pygame.mixer.init(frequency=sample_rate)
        except pygame.error as e:
            self._debug_print(f"Pygame mixer init error: {e}")
            return ("", text) if not stream else iter([])

        # Shared state containers
        self.is_playing = [True]
        self.user_interrupted = [False]
        result_spoken_text = [""]
        result_unspoken_text = [text]
        playback_start_time = [0.0]

        def get_current_position() -> float:
            if pygame.mixer.music.get_busy() and playback_start_time[0] > 0:
                return time.time() - playback_start_time[0]
            return 0.0

        def update_text_progress():
            pos_seconds = get_current_position()
            proportion = min(1.0, max(0.0, (pos_seconds / total_duration) if total_duration else 0))
            char_pos = int(len(text) * proportion)
            result_spoken_text[0] = text[:char_pos]
            result_unspoken_text[0] = text[char_pos:]

        def audio_thread_fn():
            try:
                pygame.mixer.music.load(audio_buffer)
                pygame.mixer.music.set_volume(0.0 if mute else 1.0)
                playback_start_time[0] = time.time()
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not self.user_interrupted[0]:
                    time.sleep(0.05)
                if self.user_interrupted[0]:
                    pygame.mixer.music.stop()
                else:
                    # Force completion state
                    result_spoken_text[0] = text
                    result_unspoken_text[0] = ""
            except Exception as e:
                self._debug_print(f"Audio playback error: {e}")
            finally:
                self.is_playing[0] = False

        at = threading.Thread(target=audio_thread_fn, daemon=True)
        at.start()

        # Non-streaming path (no generator)
        if not stream:
            last_callback_time = 0.0
            while self.is_playing[0]:
                update_text_progress()
                now = time.time()
                if self.playback_position_callback and now - last_callback_time > 0.05:
                    self.playback_position_callback(get_current_position(), True)
                    last_callback_time = now
                # Removed keyboard space interruption logic
                time.sleep(0.02)
            if self.playback_position_callback:
                self.playback_position_callback(0.0, False)
            at.join(timeout=1.0)
            return result_spoken_text[0], result_unspoken_text[0]

        # Streaming path (return generator)
        def stream_generator():
            last_spoken = ""
            while self.is_playing[0]:
                update_text_progress()
                spoken = result_spoken_text[0]

                if self.playback_position_callback:
                    self.playback_position_callback(get_current_position(), True)

                # Removed keyboard interruption logic

                chunk = ""
                if spoken != last_spoken:
                    chunk = spoken[len(last_spoken):]
                    if self.text_stream_callback and chunk:
                        self.text_stream_callback(chunk)
                    last_spoken = spoken

                yield chunk
                time.sleep(0.02)

            if result_spoken_text[0] != last_spoken:
                final_chunk = result_spoken_text[0][len(last_spoken):]
                if self.text_stream_callback and final_chunk:
                    self.text_stream_callback(final_chunk)
                if final_chunk:
                    yield final_chunk

            if self.playback_position_callback:
                self.playback_position_callback(0.0, False)
            at.join(timeout=1.0)

        return stream_generator()

    def stop_playback(self):
        """Stop current audio playback."""
        self.user_interrupted[0] = True
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()

    def close(self) -> None:
        """Gracefully shutdown TTS engine resources."""
        try:
            self._debug_print("Shutting down TTS engine...")
            
            # Stop any ongoing audio playback
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                self._debug_print("Pygame mixer shutdown complete")
            
            # Clean up voice model resources
            if hasattr(self, 'voice') and self.voice is not None:
                del self.voice
                self._debug_print("Voice model resources cleaned up")
            
            self.logging_manager.add_message("TTSEngine shutdown complete", level="INFO", source="TTSEngine")
            
        except Exception as e:
            self._debug_print(f"Error during TTS engine shutdown: {e}")
            self.logging_manager.add_message(f"Error during TTSEngine shutdown: {e}", level="ERROR", source="TTSEngine")


# ASREngine and SpeechEngine classes remain the same
class ASREngine:
    """Automatic Speech Recognition engine using Whisper."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize the ASR engine.
        
        Args:
            debug (bool): Enable debug logging
        """
        self.logging_manager = LoggingManager()
        self.logging_manager.add_message("Initiating - ASREngine", level="INFO", source="ASREngine")
        
        self.debug = debug
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
        
        self._debug_print("ASR Engine initialized")

    def _debug_print(self, message: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(f"[ASR DEBUG] {message}")

    def record_and_transcribe(self) -> Optional[str]:
        """
        Records and transcribes user speech input.
        NOTE: currently recording will get triggered from my_ai_assistant.py
        Args:
            duration (int): Maximum recording duration in seconds. Defaults to 5.

        Returns:
            Optional[str]: Transcribed text or None if transcription fails
        """
        self.logging_manager.add_message(f"Recording speech input.", level="INFO", source="ASREngine")
        self._debug_print(f"Starting recording")
        audio_queue = queue.Queue()
        frames = []

        def audio_callback(in_data, frame_count, time_info, status):
            audio_queue.put(in_data)
            return (None, pyaudio.paContinue)

        stream = None
        try:
            while keyboard.is_pressed("space"):
                if stream is None:
                    self._debug_print("Opening audio stream")
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
                self._debug_print("Closing audio stream")
                stream.stop_stream()
                stream.close()

        if len(frames) > 0:
            self._debug_print(f"Recorded {len(frames)} frames, saving to file")
            
            # Save the recorded audio to a temporary WAV file
            with wave.open(self.WAVE_OUTPUT_FILENAME, "wb") as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b"".join(frames))

            try:
                self._debug_print("Starting transcription")
                result = self.engine.transcribe(media=self.WAVE_OUTPUT_FILENAME)
                
                # Clean up temporary file
                if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                    os.remove(self.WAVE_OUTPUT_FILENAME)
                    
                transcribed_text = result[0].text if result else None
                self._debug_print(f"Transcription result: '{transcribed_text}'")
                return transcribed_text

            except Exception as e:
                self._debug_print(f"Transcription error: {e}")
                if os.path.exists(self.WAVE_OUTPUT_FILENAME):
                    os.remove(self.WAVE_OUTPUT_FILENAME)
                return None
        else:
            self._debug_print("No audio frames recorded")
            return None

    def close(self) -> None:
        """Gracefully shutdown ASR engine resources."""
        try:
            self._debug_print("Shutting down ASR engine...")
            
            # Clean up PyAudio resources
            if hasattr(self, 'pyaudio') and self.pyaudio is not None:
                self.pyaudio.terminate()
                self._debug_print("PyAudio terminated")
            
            # Clean up Whisper model resources
            if hasattr(self, 'engine') and self.engine is not None:
                del self.engine
                self._debug_print("Whisper model resources cleaned up")
            
            # Clean up temporary audio file if it exists
            if hasattr(self, 'WAVE_OUTPUT_FILENAME') and os.path.exists(self.WAVE_OUTPUT_FILENAME):
                os.remove(self.WAVE_OUTPUT_FILENAME)
                self._debug_print("Temporary audio file cleaned up")
            
            self.logging_manager.add_message("ASREngine shutdown complete", level="INFO", source="ASREngine")
            
        except Exception as e:
            self._debug_print(f"Error during ASR engine shutdown: {e}")
            self.logging_manager.add_message(f"Error during ASREngine shutdown: {e}", level="ERROR", source="ASREngine")


class SpeechEngine:
    """Main speech engine combining TTS and ASR functionality."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize the speech engine.
        
        Args:
            debug (bool): Enable debug logging for all components
        """
        self.logging_manager = LoggingManager()
        self.logging_manager.add_message("Initiating - SpeechEngine", level="INFO", source="SpeechEngine")
        
        self.debug = debug
        self.tts_engine = TTSEngine(debug=debug)
        self.asr_engine = ASREngine(debug=debug)
        
        if self.debug:
            print("[SPEECH ENGINE DEBUG] Speech engine initialized")

    
    
    def speak(self, text: str, stream: bool = False, visualize: bool = True, mute: bool = False) -> Union[Tuple[str, str], Generator[str, None, None]]:
        """
        Speak the given text using TTS.
        
        Args:
            text (str): Text to speak
            stream (bool): Whether to stream text character by character
            visualize (bool): Whether to show LED visualization
            mute (bool): Whether to mute audio output
            
        Returns:
            Union[Tuple[str, str], Generator[str, None, None]]: 
                If stream=True: Generator yielding characters
                If stream=False: Tuple of (spoken_text, unspoken_text)
        """
        if self.debug:
            print(f"[SPEECH ENGINE DEBUG] Speaking text - Stream: {stream}, Visualize: {visualize}, Mute: {mute}; Text: '{text}...'")
        try:
            x = self.tts_engine.play_synthesized_audio(
                text, stream=stream, mute=mute
            ) # visualize=visualize,
        except Exception as e:
            if self.debug:
                print(f"[SPEECH ENGINE DEBUG] Error during speak: {e}")
            self.logging_manager.add_message(f"Error during speak: {e}", level="ERROR", source="SpeechEngine")
            return ("", text) if not stream else iter([])
        return x
        
    
    def listen(self, duration: int = 5) -> Optional[str]:
        """
        Listen for speech input and transcribe it.
        
        Args:
            duration (int): Maximum recording duration in seconds
            
        Returns:
            Optional[str]: Transcribed text or None if transcription fails
        """
        if self.debug:
            print(f"[SPEECH ENGINE DEBUG] Listening for speech")
        
        return self.asr_engine.record_and_transcribe()
    
    def close(self) -> None:
        """Gracefully shutdown all speech engine resources."""
        try:
            if self.debug:
                print("[SPEECH ENGINE DEBUG] Shutting down speech engine...")
            
            # Close TTS engine
            if hasattr(self, 'tts_engine') and self.tts_engine is not None:
                self.tts_engine.close()
            
            # Close ASR engine
            if hasattr(self, 'asr_engine') and self.asr_engine is not None:
                self.asr_engine.close()
            
            self.logging_manager.add_message("SpeechEngine shutdown complete", level="INFO", source="SpeechEngine")
            
            if self.debug:
                print("[SPEECH ENGINE DEBUG] Speech engine shutdown complete")
                
        except Exception as e:
            if self.debug:
                print(f"[SPEECH ENGINE DEBUG] Error during speech engine shutdown: {e}")
            self.logging_manager.add_message(f"Error during SpeechEngine shutdown: {e}", level="ERROR", source="SpeechEngine")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()
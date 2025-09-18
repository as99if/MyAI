"""
@author - Aisf Ahmed - asif.shuvo2199@outlook.com
"""
import os
import io
import queue
from typing import Optional, Tuple, Generator, Union
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
    """Text-to-Speech engine with optional LED visualization."""
    
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
        
        # Visualization parameters
        self.screen = None
        self.screen_width = 600
        self.screen_height = 600
        self.bg_color = (10, 10, 10)
        self.num_bands = 32
        self.led_levels = 32

        # Calculate responsive bar dimensions
        available_width = int(self.screen_width * 0.85)
        self.bar_spacing = max(2, int(self.screen_width * 0.007))
        self.bar_width = max(1, (available_width - (self.num_bands - 1) * self.bar_spacing) // self.num_bands)
        
        self.led_colors = [(255, 255, 0)] * self.led_levels
        self.user_interrupted = [False]
        pygame.init()
        
    def _debug_print(self, message: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(f"[TTS DEBUG] {message}")
        
    def init_screen(self) -> None:
        """Initialize the pygame screen for visualization."""
        if self.screen is None:
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("🎵 LED Spectrum Analyzer 🎵")
            self.screen.fill(self.bg_color)
            pygame.display.flip()
            self.margin_x = 20
            self.margin_y = 20
            self._debug_print("Screen initialized")
        else:
            self.screen.fill(self.bg_color)
    
    def get_screen(self) -> Optional[pygame.Surface]:
        """
        Get the pygame screen surface.
        
        Returns:
            Optional[pygame.Surface]: The screen surface or None if not initialized
        """
        return self.screen
        
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

    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        """
        Wrap text to fit within the given width.
        
        Args:
            text (str): Text to wrap
            font (pygame.font.Font): Font to use for measuring text
            max_width (int): Maximum width in pixels
            
        Returns:
            list[str]: List of wrapped text lines
        """
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def play_synthesized_audio_with_or_without_visualizer(
        self, 
        text: str, 
        stream: bool = False, 
        visualize: bool = True, 
        mute: bool = False
    ) -> Union[Tuple[str, str], Generator[str, None, None]]:
        """
        Play synthesized audio with optional visualization and streaming.
        
        Args:
            text (str): Text to synthesize and play
            stream (bool): Whether to stream text character by character
            visualize (bool): Whether to show LED visualization
            mute (bool): Whether to mute audio output
            
        Returns:
            Union[Tuple[str, str], Generator[str, None, None]]: 
                If stream=True: Generator yielding characters
                If stream=False: Tuple of (spoken_text, unspoken_text)
        """
        _parameters = ""
        if stream:
            _parameters += "Stream: True; "
        if visualize:
            _parameters += "Visualize: True; "
        if mute:
            _parameters += "Mute: True; "
        self.logging_manager.add_message(f"Synthesizing reply - {_parameters}", level="INFO", source="TTSEngine")
        
        self._debug_print(f"Playing audio - Stream: {stream}, Visualize: {visualize}, Mute: {mute}")
        
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

        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=sample_rate)
        
        # Audio playback state
        is_playing = [True]
        self.user_interrupted = [False]
        result_spoken_text = [""]
        result_unspoken_text = [""]

        def get_current_position() -> float:
            """Get current audio playback position in seconds."""
            if pygame.mixer.music.get_busy():
                return pygame.mixer.music.get_pos() / 1000.0
            return 0

        def calculate_text_split() -> float:
            """
            Calculate the text split position based on current audio playback progress.
            
            Returns:
                float: The proportion of audio that has been played (0.0 to 1.0)
            """
            pos_seconds = get_current_position()
            total_duration = total_frames / sample_rate
            proportion_played = min(1.0, max(0.0, pos_seconds / total_duration))
            char_position = int(len(text) * proportion_played)
            result_spoken_text[0] = text[:char_position]
            result_unspoken_text[0] = text[char_position:]
            return proportion_played

        def play_audio() -> None:
            """Audio playback thread function."""
            pygame.mixer.music.load(audio_buffer)
            volume = 0.0 if mute else 1.0
            pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.play()
            
            try:
                while pygame.mixer.music.get_busy() and not self.user_interrupted[0]:
                    time.sleep(0.01)
                if self.user_interrupted[0]:
                    self._debug_print("Audio thread detected interruption")
                else:
                    result_spoken_text[0] = text
                    result_unspoken_text[0] = ""
                    self._debug_print("Playback completed normally")
            except Exception as e:
                self._debug_print(f"Error in audio thread: {e}")
            finally:
                is_playing[0] = False

        # Start audio playback
        audio_thread = threading.Thread(target=play_audio)
        audio_thread.daemon = True
        audio_thread.start()

        # Handle case where visualization is disabled
        if not visualize:
            last_streamed_spoken = ""
            
            while is_playing[0]:
                def on_spacebar():
                    if not self.user_interrupted[0]:
                        self.user_interrupted[0] = True
                        self._debug_print("🛑 Playback interrupted by SPACEBAR")
                        if pygame.mixer.get_init():
                            pygame.mixer.music.stop()
                
                keyboard.on_press_key('space', lambda _: on_spacebar())
                
                if self.user_interrupted[0]:
                    self._debug_print("Audio thread detected interruption")
                    pygame.mixer.music.stop()
                    is_playing[0] = False
                    break
                    
                calculate_text_split()
                spoken = result_spoken_text[0]
                
                # Stream spoken text updates
                if stream and spoken != last_streamed_spoken:
                    if len(spoken) > len(last_streamed_spoken):
                        for i in range(len(last_streamed_spoken), len(spoken)):
                            yield spoken[i]
                    last_streamed_spoken = spoken
                
                time.sleep(0.01)
            
            # Ensure we yield any remaining characters after playback completes
            if stream:
                final_spoken = result_spoken_text[0]
                if len(final_spoken) > len(last_streamed_spoken):
                    for i in range(len(last_streamed_spoken), len(final_spoken)):
                        yield final_spoken[i]
            
            return result_spoken_text[0], result_unspoken_text[0]

        # Visualization code continues...
        self._debug_print("Starting visualization")
        self.init_screen()
        pygame.display.set_caption("🎵 LED Spectrum Analyzer - Press SPACE to interrupt 🎵")
        
        # Calculate font sizes based on screen dimensions
        font_size = max(16, int(self.screen_height * 0.003))
        small_font_size = max(12, int(self.screen_height * 0.003))
        font = pygame.font.SysFont("consolas", font_size)
        small_font = pygame.font.SysFont("consolas", small_font_size)

        # Analyzer setup
        num_bands = self.num_bands
        led_levels = self.led_levels
        chunk_duration = 0.05
        chunk_size = int(sample_rate * chunk_duration)
        freq_bands = np.logspace(np.log10(20), np.log10(sample_rate//2), num_bands + 1)
        
        min_freq = 128
        start_band = next((i for i, f in enumerate(freq_bands) if f > min_freq), 0)
        band_centers = (freq_bands[:-1] + freq_bands[1:]) / 2
        smoothing_factor = 0.3
        previous_levels = np.zeros(num_bands)

        # Main visualization loop
        running = True
        clock = pygame.time.Clock()
        interrupted = False
        info_message = "Press SPACEBAR to interrupt playback"
        spoken_color = (0, 255, 0)
        unspoken_color = (255, 0, 0)
        last_streamed_spoken = ""

        while running:
            self.screen.fill(self.bg_color)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.user_interrupted[0] = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not interrupted:
                    self.user_interrupted[0] = True
                    interrupted = True
                    proportion_played = calculate_text_split()
                    pygame.mixer.music.stop()
                    info_message = f"🛑 Playback Interrupted at {proportion_played:.1%} 🛑"
                    self._debug_print(f"Spoken text: '{result_spoken_text[0]}'")
                    self._debug_print(f"Unspoken text: '{result_unspoken_text[0]}'")

            # Synchronize visualization with audio playback
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

            # Draw LED bars
            for i, level in enumerate(previous_levels):
                lit_leds = int(level * led_levels - 6)
                for j in range(led_levels):
                    margin_left = int(self.screen_width * 0.067)
                    margin_bottom = int(self.screen_height * 0.067)
                    visualizer_height = int(self.screen_height * 0.8)
                    
                    x = margin_left + i * (self.bar_width + self.bar_spacing)
                    y = self.screen_height - margin_bottom - (j + 1) * visualizer_height // led_levels
                    
                    bar_height = max(2, visualizer_height // led_levels - 2)
                    rect = pygame.Rect(x, y, self.bar_width, bar_height)
                    
                    if j < lit_leds:
                        intensity = min(1.0, (j + 1) / lit_leds) if lit_leds > 0 else 0
                        color = (255, 255, int(80 + 175 * intensity))
                        pygame.draw.rect(self.screen, color, rect, border_radius=2)
                        pygame.draw.rect(self.screen, (255, 255, 255), rect, 1, border_radius=2)
                    else:
                        pygame.draw.rect(self.screen, (60, 60, 20), rect, border_radius=2)
                        pygame.draw.rect(self.screen, (30, 30, 30), rect, 1, border_radius=2)

            # Draw frequency labels
            for i in range(0, num_bands, 4):
                margin_left = int(self.screen_width * 0.067)
                label_y = int(self.screen_height * 0.95)
                
                x = margin_left + i * (self.bar_width + self.bar_spacing) + self.bar_width // 2
                freq = band_centers[i]
                if freq < 1000:
                    label = f"{int(freq)}Hz"
                else:
                    label = f"{freq/1000:.1f}kHz"
                label_surface = small_font.render(label, True, (200, 200, 200))
                self.screen.blit(label_surface, (x - label_surface.get_width() // 2, label_y))
            
            # Draw spoken/unspoken text
            if interrupted or not is_playing[0]:
                spoken = result_spoken_text[0]
                unspoken = result_unspoken_text[0]
            else:
                proportion_played = calculate_text_split()
                spoken = result_spoken_text[0]
                unspoken = result_unspoken_text[0]
            
            # Stream spoken text updates
            # this will be used by the ui thread to update the chat window in real-time, not the response from process_and_create_chat_generation
            if stream and spoken != last_streamed_spoken:
                if len(spoken) > len(last_streamed_spoken):
                    # last letter spoken
                    # for i in range(len(last_streamed_spoken), len(spoken)):
                    #     yield spoken[i]
                    last_word_spoken = spoken[len(last_streamed_spoken):]
                    yield last_word_spoken
                last_streamed_spoken = spoken
            
            # Wrap text if it's too long for the screen
            max_text_width = self.screen_width - 2 * self.margin_x
            spoken_lines = self._wrap_text(spoken, font, max_text_width)
            unspoken_lines = self._wrap_text(unspoken, font, max_text_width)
            
            text_y = self.margin_y
            for i, line in enumerate(spoken_lines[:6]):
                spoken_surface = font.render(line, True, spoken_color)
                self.screen.blit(spoken_surface, (self.margin_x, text_y + i * font_size))

            remaining_lines = 3 - len(spoken_lines)
            if remaining_lines > 0 and unspoken_lines:
                for i, line in enumerate(unspoken_lines[:remaining_lines]):
                    unspoken_surface = font.render(line, True, unspoken_color)
                    y_pos = text_y + (len(spoken_lines) + i) * font_size
                    self.screen.blit(unspoken_surface, (self.margin_x, y_pos))

            # Draw info message
            info_surface = small_font.render(info_message, True, (0, 255, 255))
            self.screen.blit(info_surface, (40, self.screen_height - 60))

            pygame.display.flip()
            clock.tick(60)

            if not is_playing[0] and not interrupted:
                info_message = "Playback completed"
                interrupted = True

        pygame.quit()
        self._debug_print("Visualization window closed")
        self._debug_print(f"Final results - Spoken: {len(result_spoken_text[0])} chars, Unspoken: {len(result_unspoken_text[0])} chars")
        return result_spoken_text[0], result_unspoken_text[0]

    def close(self) -> None:
        """Gracefully shutdown TTS engine resources."""
        try:
            self._debug_print("Shutting down TTS engine...")
            
            # Stop any ongoing audio playback
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                self._debug_print("Pygame mixer shutdown complete")
            
            # Quit pygame if initialized
            if pygame.get_init():
                pygame.quit()
                self._debug_print("Pygame shutdown complete")
            
            # Clean up voice model resources
            if hasattr(self, 'voice') and self.voice is not None:
                del self.voice
                self._debug_print("Voice model resources cleaned up")
            
            self.logging_manager.add_message("TTSEngine shutdown complete", level="INFO", source="TTSEngine")
            
        except Exception as e:
            self._debug_print(f"Error during TTS engine shutdown: {e}")
            self.logging_manager.add_message(f"Error during TTSEngine shutdown: {e}", level="ERROR", source="TTSEngine")


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
            print(f"[SPEECH ENGINE DEBUG] Speaking text - Stream: {stream}, Visualize: {visualize}, Mute: {mute}")
        
        return self.tts_engine.play_synthesized_audio_with_or_without_visualizer(
            text, stream=stream, visualize=visualize, mute=mute
        )
    
    
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
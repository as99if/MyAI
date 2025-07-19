"""
@author - Aisf Ahmed - asif.shuvo2199@outlook.com
"""


import multiprocessing
import os

from pywhispercpp.model import Model as Whisper_ccp
import wave
import io
import pygame
import numpy as np
import threading
import time
from scipy.fft import fft
from scipy.signal import get_window
from piper import PiperVoice, SynthesisConfig

    

class TTSEngine:
    def __init__(self, model_path="/home/asifahmedshuvo/Development/MyAI/my_ai_assistant/models/speech/en_GB-semaine-medium.onnx", screen=None):
        self.voice = PiperVoice.load(model_path)
        self.syn_config = SynthesisConfig(
            volume=0.5,
            length_scale=1.0,
            noise_scale=1.0,
            noise_w_scale=1.0,
            normalize_audio=False,
        )
        # Visualization parameters
        self.screen = screen
        self.screen_width = 1800
        self.screen_height = 700
        self.bg_color = (10, 10, 10)
        self.num_bands = 32
        self.led_levels = 32
        self.bar_width = self.screen_width // (self.num_bands + 2)
        self.bar_spacing = 4
        self.led_colors = [(255, 255, 0)] * self.led_levels  # Yellow LEDs
        # self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        # self.init_screen()
        self.user_interrupted = [False]
        pygame.init()
        
    def init_screen(self):
        if self.screen is None:
            font = pygame.font.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("🎵 LED Spectrum Analyzer 🎵")
            self.screen.fill(self.bg_color)
            pygame.display.flip()
        else:
            self.screen.fill(self.bg_color)
    
    def set_screen(self, screen):
        self.screen = screen
    
    def get_screen(self):
        if self.screen is None:
            return None
        return self.screen
        
    def synthesize(self, text):
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file=wav_file, syn_config=self.syn_config)
        audio_buffer.seek(0)
        return audio_buffer

    def play_synthesized_audio_with_led_visualizer(self, text) -> tuple:
        """
        Play synthesized audio with LED visualizer and interruption support. 
        Audio is synthesized in this methodfrom the given text and visualized in real-time using a spectrum analyzer with LED bars.

        Args:
            text (str): Text to be synthesized and visualized.

        Returns:
            tuple (str, str): A tuple containing the spoken and unspoken parts of the text.
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

        # pygame.init()
        pygame.mixer.init(frequency=sample_rate)
        # screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("🎵 LED Spectrum Analyzer - Press SPACE to interrupt 🎵")
        font = pygame.font.SysFont("consolas", 22)
        small_font = pygame.font.SysFont("consolas", 16)

        # Analyzer setup
        num_bands = self.num_bands
        led_levels = self.led_levels
        chunk_duration = 0.05
        chunk_size = int(sample_rate * chunk_duration)
        freq_bands = np.logspace(np.log10(20), np.log10(sample_rate//2), num_bands + 1)
        # print("-----")
        # print(len(freq_bands), freq_bands)
        # print("-----")
        
        # Find the first band above 128 Hz
        min_freq = 128
        start_band = next((i for i, f in enumerate(freq_bands) if f > min_freq), 0)
        band_centers = (freq_bands[:-1] + freq_bands[1:]) / 2
        smoothing_factor = 0.3
        previous_levels = np.zeros(num_bands)
        result_spoken_text = [""]
        result_unspoken_text = [""]

        # Audio playback state
        is_playing = [True]
        self.user_interrupted = [False]
        current_chunk = [0]

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

        # Main visualization loop
        running = True
        clock = pygame.time.Clock()
        interrupted = False
        info_message = "Press SPACEBAR to interrupt playback"
        spoken_color = (0, 255, 0)
        unspoken_color = (255, 0, 0)
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
                    print("\nResults:")
                    print(f"Spoken text: '{result_spoken_text[0]}'")
                    print(f"Unspoken text: '{result_unspoken_text[0]}'")

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

            # Draw LED bars
            for i, level in enumerate(previous_levels):
                lit_leds = int(level * led_levels - 6) # control led level hight
                for j in range(led_levels):
                    x = 40 + i * (self.bar_width + self.bar_spacing)
                    y = self.screen_height - 40 - (j + 1) * (self.screen_height - 120) // led_levels
                    rect = pygame.Rect(x, y, self.bar_width, (self.screen_height - 120) // led_levels - 2)
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
                x = 40 + i * (self.bar_width + self.bar_spacing) + self.bar_width // 2
                freq = band_centers[i]
                if freq < 1000:
                    label = f"{int(freq)}Hz"
                else:
                    label = f"{freq/1000:.1f}kHz"
                label_surface = small_font.render(label, True, (200, 200, 200))
                self.screen.blit(label_surface, (x - label_surface.get_width() // 2, self.screen_height - 30))

            # Draw spoken/unspoken text
            if interrupted or not is_playing[0]:
                spoken = result_spoken_text[0]
                unspoken = result_unspoken_text[0]
            else:
                proportion_played = calculate_text_split()
                spoken = result_spoken_text[0]
                unspoken = result_unspoken_text[0]
            spoken_surface = font.render(spoken, True, spoken_color)
            unspoken_surface = font.render(unspoken, True, unspoken_color)
            self.screen.blit(spoken_surface, (40, 20))
            self.screen.blit(unspoken_surface, (40 + spoken_surface.get_width() + 10, 20))

            # Draw info message
            info_surface = small_font.render(info_message, True, (0, 255, 255))
            self.screen.blit(info_surface, (40, self.screen_height - 60))

            pygame.display.flip()
            
            clock.tick(60)

            if not is_playing[0] and not interrupted:
                info_message = "Playback completed"
                interrupted = True

        pygame.quit()
        print("Visualization window closed")
        print(f"Final results - Spoken: {len(result_spoken_text[0])} chars, Unspoken: {len(result_unspoken_text[0])} chars")
        return result_spoken_text[0], result_unspoken_text[0]

    def play_synthesized_audio_without_led_visualizer(self, text):
        # TODO: fix interruption support
        
        audio_buffer = self.synthesize(text)
        with wave.open(audio_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            total_frames = wav_file.getnframes()
        
        pygame.mixer.init(frequency=sample_rate)

        result_spoken_text = [""]
        result_unspoken_text = [""]
        is_playing = [True]
        self.user_interrupted = [False]

        audio_buffer.seek(0)

        # pygame.init()
        pygame.mixer.init(frequency=sample_rate)
        
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

        running = True
        interrupted = False
        info_message = "Press SPACEBAR to interrupt playback"
        while running:
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
                    print("\nResults:")
                    print(f"Spoken text: '{result_spoken_text[0]}'")
                    print(f"Unspoken text: '{result_unspoken_text[0]}'")
            time.sleep(0.01)
            if not is_playing[0] and not interrupted:
                info_message = "Playback completed"
                interrupted = True
            if not is_playing[0]:
                running = False

        pygame.quit()
        print("Audio window closed")
        print(f"Final results - Spoken: {len(result_spoken_text[0])} chars, Unspoken: {len(result_unspoken_text[0])} chars")
        return result_spoken_text[0], result_unspoken_text[0]


class ASREngine:
    def __init__(self):
        
        # self.engine = whisper.load_model("medium.en", download_root="./models/speech/", in_memory=False)
        self.engine = Whisper_ccp(
            model="large-v3-turbo", # large-v3, large-v3-turbo-q5_0, large-v3-turbo-q5_0, medium.en, large-v3
            models_dir="./models/speech/",
            print_realtime=False,
            print_progress=False
        )
        

    

class SpeechEngine:
    
    def __init__(self):
        self.tts_engine = TTSEngine()
        self.asr_engine = ASREngine()


def test_tts():
    tts_engine = TTSEngine()
    # tts_engine.init_screen()
    time.sleep(1)
    print('🎵 Starting LED Spectrum Analyzer with interruption support...')
    print('Press SPACEBAR to interrupt playback')
    text = "Welcome to the world of speech synthesis! This is a demonstration of text-to-speech with a real-time audio visualizer. You can press the spacebar at any time to interrupt the playback."
    # spoken, unspoken = tts_engine.play_synthesized_audio_without_led_visualizer(text)

    tts_engine.init_screen()
    spoken, unspoken = tts_engine.play_synthesized_audio_with_led_visualizer(text)

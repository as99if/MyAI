import wave
import io
import pygame
import numpy as np
import threading
import time
from scipy.fft import fft
from scipy.signal import get_window
from piper import PiperVoice, SynthesisConfig
import keyboard


class TTSEngine:
    def __init__(self, model_path="/Users/asifahmed/personal/development/MyAI/my_ai_assistant/models/speech/en_GB-semaine-medium.onnx"):
       
        
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

        # self.bar_width = self.screen_width // (self.num_bands + 2)
        # self.bar_spacing = 4
        # Calculate responsive bar dimensions
        available_width = int(self.screen_width * 0.85)  # Use 85% of screen width
        self.bar_spacing = max(2, int(self.screen_width * 0.007))  # ~4px for 600px width
        self.bar_width = max(1, (available_width - (self.num_bands - 1) * self.bar_spacing) // self.num_bands)
        

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
            # draw initial background with unlid led's on thebars
            self.margin_x = 20
            self.margin_y = 20
        else:
            self.screen.fill(self.bg_color)
    
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

    def _wrap_text(self, text, font, max_width):
        """Wrap text to fit within the given width"""
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
    
    def play_synthesized_audio_with_or_without_visualizer(self, text: str, stream: bool=False, visualize: bool=True, mute: bool=False):
        # # just strem text without audio or visualization
        # if not visualize:
        #     # Only yield text character by character without audio/visualization
        #     if stream:
        #         for char in text:
        #             yield char
        #     return text, ""

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

        def get_current_position():
            if pygame.mixer.music.get_busy():
                return pygame.mixer.music.get_pos() / 1000.0
            return 0

        def calculate_text_split() -> float:
            """
            Calculate the text split position based on current audio playback progress.
            
            This method synchronizes text display with audio playback by determining
            how much of the text should be marked as "spoken" versus "unspoken" based
            on the current playback position in the audio stream.
            
            Returns:
                float: The proportion of audio that has been played, clamped between 0.0 and 1.0.
            """
            pos_seconds = get_current_position()
            total_duration = total_frames / sample_rate
            proportion_played = min(1.0, max(0.0, pos_seconds / total_duration))
            char_position = int(len(text) * proportion_played)
            result_spoken_text[0] = text[:char_position]
            result_unspoken_text[0] = text[char_position:]
            return proportion_played

        def play_audio():
            pygame.mixer.music.load(audio_buffer)
            # Set volume based on audio_on parameter
            volume = 0.0 if mute else 1.0
            pygame.mixer.music.set_volume(volume)
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

        # Start audio playback
        
        audio_thread = threading.Thread(target=play_audio)
        audio_thread.daemon = True
        audio_thread.start()

        # Handle case where visualization is disabled
        if not visualize:
            last_streamed_spoken = ""
            
            while is_playing[0]:
                # Handle keyboard events to detect SPACEBAR interrupt
                def on_spacebar():
                    if not self.user_interrupted[0]:  # Only interrupt if not already interrupted
                        self.user_interrupted[0] = True
                        print("🛑 Playback interrupted by SPACEBAR")
                        if pygame.mixer.get_init():
                            pygame.mixer.music.stop()
                
                # Register spacebar handler
                keyboard.on_press_key('space', lambda _: on_spacebar())
                # Check if already interrupted from event handling
                if self.user_interrupted[0]:
                    print("Audio thread detected interruption")
                    pygame.mixer.music.stop()
                    is_playing[0] = False
                    break
                calculate_text_split()
                spoken = result_spoken_text[0]
                
                # Stream spoken text updates
                if stream and spoken != last_streamed_spoken:
                    if len(spoken) > len(last_streamed_spoken):
                        # Yield new characters one by one
                        for i in range(len(last_streamed_spoken), len(spoken)):
                            yield spoken[i]
                    last_streamed_spoken = spoken
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
            
            # Ensure we yield any remaining characters after playback completes
            if stream:
                final_spoken = result_spoken_text[0]
                if len(final_spoken) > len(last_streamed_spoken):
                    for i in range(len(last_streamed_spoken), len(final_spoken)):
                        yield final_spoken[i]
            
            return result_spoken_text[0], result_unspoken_text[0]

        # Original visualization code continues here...
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
        
        # Find the first band above 128 Hz
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
                lit_leds = int(level * led_levels - 6) # control led level height
                for j in range(led_levels):
                    # Calculate positions based on screen dimensions
                    margin_left = int(self.screen_width * 0.067)  # ~40px for 600px width
                    margin_bottom = int(self.screen_height * 0.067)  # ~40px for 600px height
                    visualizer_height = int(self.screen_height * 0.8)  # 80% of screen height for visualizer
                    
                    x = margin_left + i * (self.bar_width + self.bar_spacing)
                    y = self.screen_height - margin_bottom - (j + 1) * visualizer_height // led_levels
                    
                    bar_height = max(2, visualizer_height // led_levels - 2)  # Ensure minimum height
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
                label_y = int(self.screen_height * 0.95)  # Position labels at 95% of screen height
                
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
            if stream and spoken != last_streamed_spoken:
                last_word_spoken = spoken[len(last_streamed_spoken):]
                last_letter_spoken = spoken[-1] if spoken else ""

                yield last_letter_spoken
                last_streamed_spoken = spoken
                
            # Wrap text if it's too long for the screen
            max_text_width = self.screen_width - 2 * self.margin_x
            spoken_lines = self._wrap_text(spoken, font, max_text_width)
            unspoken_lines = self._wrap_text(unspoken, font, max_text_width)
            
            text_y = self.margin_y
            for i, line in enumerate(spoken_lines[:6]):  # Limit to 6 lines
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
        print("Visualization window closed")
        print(f"Final results - Spoken: {len(result_spoken_text[0])} chars, Unspoken: {len(result_unspoken_text[0])} chars")
        return result_spoken_text[0], result_unspoken_text[0]

    

# TEST pygame app
def run_pygame_app():
    tts_engine = TTSEngine()
    # tts_engine.init_screen()
    time.sleep(1)
    print('🎵 Starting LED Spectrum Analyzer with interruption support...')
    print('Press SPACEBAR to interrupt playback')
    text = "Welcome to the world of speech synthesis! This is a demonstration of text-to-speech with a real-time audio visualizer. You can press the spacebar at any time to interrupt the playback."
    # spoken, unspoken = tts_engine.play_synthesized_audio_without_led_visualizer(text)

    tts_engine.init_screen()
    # spoken, unspoken = tts_engine.play_synthesized_audio_with_or_without_led_visualizer(text, stream=False, visualize=True, mute=False)
    
    # Use as generator
    # for spoken_text in tts_engine.play_synthesized_audio_with_or_without_led_visualizer(text, stream=True, visualize=True, mute=False):
    #     print(f"Streamed: {spoken_text}")
    
    # Use as generator, no visualization UI or playback - just text stream
    for spoken_text in tts_engine.play_synthesized_audio_with_or_without_visualizer(text, stream=True, visualize=False, mute=True):
        print(f"Streamed: {spoken_text}")
        time.sleep(0.03) # Simulate processing delay

    
run_pygame_app()
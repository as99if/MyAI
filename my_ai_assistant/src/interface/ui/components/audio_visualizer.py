"""
Audio Visualizer Component

This module implements an audio spectrum analyzer visualizer component that can be
integrated into the MyAI UI system.
"""

import numpy as np
import pygame
import threading
import time
from typing import Optional
from scipy.fft import fft
from scipy.signal import get_window

from .base_component import BaseComponent


class AudioVisualizer(BaseComponent):
    """
    Audio spectrum analyzer visualizer component with LED-style bars.
    """
    
    def __init__(self, rect, font, colors, padding=10):
        """
        Initialize the audio visualizer component.
        
        Args:
            rect (pygame.Rect): The rectangle defining the component's position and size.
            font (pygame.freetype.Font): The font to use for text rendering.
            colors (dict): Dictionary of color values for the component.
            padding (int): Padding around the visualizer content.
        """
        super().__init__(rect, font, colors)
        self.padding = padding
        self.has_async_event_handler = False
        
        # Visualization parameters
        self.num_bands = 32
        self.led_levels = 32
        self.bg_color = (10, 10, 10)
        
        # Calculate responsive bar dimensions
        available_width = int(self.rect.width * 0.85)
        self.bar_spacing = max(2, int(self.rect.width * 0.007))
        self.bar_width = max(1, (available_width - (self.num_bands - 1) * self.bar_spacing) // self.num_bands)
        
        # Audio processing state
        self.audio_data = None
        self.sample_rate = None
        self.is_playing = False
        self.previous_levels = np.zeros(self.num_bands)
        self.smoothing_factor = 0.3
        
        # Playback state
        self.playback_position = 0.0
        self.total_duration = 0.0
        
        # Text display
        self.spoken_text = ""
        self.unspoken_text = ""
        self.info_message = "Audio Visualizer Ready"
        
        # Colors
        self.spoken_color = (0, 255, 0)
        self.unspoken_color = (255, 0, 0)
        self.info_color = (0, 255, 255)
        
        # Font sizes
        self.font_size = max(12, int(self.rect.height * 0.02))
        self.small_font_size = max(10, int(self.rect.height * 0.015))
        
        # Initialize frequency bands
        self._setup_frequency_bands()
    
    def _setup_frequency_bands(self):
        """Setup frequency bands for spectrum analysis."""
        if self.sample_rate is None:
            self.sample_rate = 44100  # Default sample rate
        
        self.freq_bands = np.logspace(np.log10(20), np.log10(self.sample_rate//2), self.num_bands + 1)
        self.band_centers = (self.freq_bands[:-1] + self.freq_bands[1:]) / 2
        
        # Chunk parameters for real-time analysis
        self.chunk_duration = 0.05
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
    
    def set_audio_data(self, audio_data: np.ndarray, sample_rate: int, total_duration: float):
        """
        Set the audio data for visualization.
        
        Args:
            audio_data (np.ndarray): Audio waveform data.
            sample_rate (int): Sample rate of the audio.
            total_duration (float): Total duration of the audio in seconds.
        """
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.total_duration = total_duration
        self._setup_frequency_bands()
    
    def set_playback_state(self, is_playing: bool, position: float = 0.0):
        """
        Update the playback state.
        
        Args:
            is_playing (bool): Whether audio is currently playing.
            position (float): Current playback position in seconds.
        """
        self.is_playing = is_playing
        self.playback_position = position
    
    def set_text_content(self, spoken: str, unspoken: str):
        """
        Update the text content display.
        
        Args:
            spoken (str): Text that has been spoken.
            unspoken (str): Text that hasn't been spoken yet.
        """
        self.spoken_text = spoken
        self.unspoken_text = unspoken
    
    def set_info_message(self, message: str):
        """
        Set the info message displayed at the bottom.
        
        Args:
            message (str): Info message to display.
        """
        self.info_message = message
    
    def _analyze_audio_chunk(self):
        """
        Analyze the current audio chunk for visualization.
        
        Returns:
            np.ndarray: Band levels for visualization.
        """
        if self.audio_data is None or not self.is_playing:
            return self.previous_levels * 0.8
        
        # Calculate current chunk position
        start_idx = int(self.playback_position * self.sample_rate)
        end_idx = start_idx + self.chunk_size
        
        if start_idx >= len(self.audio_data):
            return self.previous_levels * 0.8
        
        # Extract and window the audio chunk
        if end_idx > len(self.audio_data):
            chunk_data = np.zeros(self.chunk_size)
            remaining = len(self.audio_data) - start_idx
            chunk_data[:remaining] = self.audio_data[start_idx:]
        else:
            chunk_data = self.audio_data[start_idx:end_idx]
        
        # Apply windowing and FFT
        windowed_data = chunk_data * get_window('hann', len(chunk_data))
        fft_data = np.abs(fft(windowed_data))[:self.chunk_size//2]
        freqs = np.fft.fftfreq(self.chunk_size, 1/self.sample_rate)[:self.chunk_size//2]
        
        # Calculate band levels
        band_levels = np.zeros(self.num_bands)
        for i in range(self.num_bands):
            band_mask = (freqs >= self.freq_bands[i]) & (freqs < self.freq_bands[i+1])
            if np.any(band_mask):
                band_levels[i] = np.sqrt(np.mean(fft_data[band_mask]**2))
        
        # Normalize and apply logarithmic scaling
        if np.max(band_levels) > 0:
            band_levels = band_levels / np.max(band_levels)
            band_levels = np.log10(band_levels + 0.01) + 2
            band_levels = np.clip(band_levels, 0, 1)
        
        # Apply smoothing
        self.previous_levels = (self.smoothing_factor * band_levels +
                               (1 - self.smoothing_factor) * self.previous_levels)
        
        return self.previous_levels
    
    def handle_event(self, event):
        """
        Handle pygame events.
        
        Args:
            event (pygame.event.Event): The pygame event to handle.
        """
        # Handle any visualizer-specific events here
        pass
    
    def update(self):
        """Update the visualizer state."""
        # Update visualization data
        self._analyze_audio_chunk()
    
    def draw(self, screen):
        """
        Draw the audio visualizer.
        
        Args:
            screen (pygame.Surface): The screen surface to draw on.
        """
        if not self.visible:
            return
        
        # Fill background
        pygame.draw.rect(screen, self.bg_color, self.rect)
        
        # Draw LED bars
        self._draw_led_bars(screen)
        
        # Draw frequency labels
        self._draw_frequency_labels(screen)
        
        # Draw text content
        self._draw_text_content(screen)
        
        # Draw info message
        self._draw_info_message(screen)
    
    def _draw_led_bars(self, screen):
        """Draw the LED-style spectrum bars."""
        band_levels = self._analyze_audio_chunk()
        
        for i, level in enumerate(band_levels):
            lit_leds = int(level * self.led_levels - 6)
            
            for j in range(self.led_levels):
                margin_left = self.rect.x + int(self.rect.width * 0.067)
                margin_bottom = int(self.rect.height * 0.067)
                visualizer_height = int(self.rect.height * 0.6)  # Reduced to make room for text
                
                x = margin_left + i * (self.bar_width + self.bar_spacing)
                y = self.rect.y + self.rect.height - margin_bottom - (j + 1) * visualizer_height // self.led_levels
                
                bar_height = max(2, visualizer_height // self.led_levels - 2)
                rect = pygame.Rect(x, y, self.bar_width, bar_height)
                
                if j < lit_leds:
                    intensity = min(1.0, (j + 1) / lit_leds) if lit_leds > 0 else 0
                    color = (255, 255, int(80 + 175 * intensity))
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, (255, 255, 255), rect, 1)
                else:
                    pygame.draw.rect(screen, (60, 60, 20), rect)
                    pygame.draw.rect(screen, (30, 30, 30), rect, 1)
    
    def _draw_frequency_labels(self, screen):
        """Draw frequency labels below the bars."""
        for i in range(0, self.num_bands, 4):
            margin_left = self.rect.x + int(self.rect.width * 0.067)
            label_y = self.rect.y + int(self.rect.height * 0.75)
            
            x = margin_left + i * (self.bar_width + self.bar_spacing) + self.bar_width // 2
            freq = self.band_centers[i]
            
            if freq < 1000:
                label = f"{int(freq)}Hz"
            else:
                label = f"{freq/1000:.1f}kHz"
            
            # Render frequency label
            text_rect = self.font.get_rect(label)
            self.font.render_to(screen, (x - text_rect.width // 2, label_y), label, (200, 200, 200))
    
    def _draw_text_content(self, screen):
        """Draw the spoken and unspoken text."""
        # Text area
        text_y = self.rect.y + self.padding
        max_text_width = self.rect.width - 2 * self.padding
        
        # Wrap text
        spoken_lines = self.wrap_text(self.spoken_text, max_text_width)
        unspoken_lines = self.wrap_text(self.unspoken_text, max_text_width)
        
        line_height = self.font_size + 2
        current_y = text_y
        
        # Draw spoken text (green)
        for i, line in enumerate(spoken_lines[:3]):  # Limit to 3 lines
            text_rect = self.font.get_rect(line)
            self.font.render_to(screen, (self.rect.x + self.padding, current_y), line, self.spoken_color)
            current_y += line_height
        
        # Draw unspoken text (red)
        remaining_lines = 3 - len(spoken_lines)
        if remaining_lines > 0 and unspoken_lines:
            for i, line in enumerate(unspoken_lines[:remaining_lines]):
                text_rect = self.font.get_rect(line)
                self.font.render_to(screen, (self.rect.x + self.padding, current_y), line, self.unspoken_color)
                current_y += line_height
    
    def _draw_info_message(self, screen):
        """Draw the info message at the bottom."""
        info_y = self.rect.y + self.rect.height - 30
        text_rect = self.font.get_rect(self.info_message)
        self.font.render_to(screen, (self.rect.x + self.padding, info_y), self.info_message, self.info_color)
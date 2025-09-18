"""
Send button component for the UI.
"""

import pygame
from .base_component import BaseComponent

class SendButton(BaseComponent):
    """
    Simple send button component.
    """
    
    def __init__(self, rect, font, colors, text=">"):
        """
        Initialize the send button component.
        
        Args:
            rect (pygame.Rect): The rectangle defining the button's position and size.
            font (pygame.freetype.Font): The font to use for text rendering.
            colors (dict): Dictionary of color values.
            text (str): The button text.
        """
        super().__init__(rect, font, colors)
        self.text = text
        self.callback = None
        self.pressed = False
        self.has_async_event_handler = False
    
    def set_click_callback(self, callback):
        """
        Set callback function to call when button is clicked.
        
        Args:
            callback (callable): Function to call when clicked.
        """
        self.callback = callback
    
    def handle_event(self, event):
        """
        Handle mouse click events.
        
        Args:
            event (pygame.event.Event): The pygame event to handle.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
                if self.callback:
                    self.callback()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.pressed = False
    
    def draw(self, screen):
        """
        Draw the send button component.
        
        Args:
            screen (pygame.Surface): The screen surface to draw on.
        """
        if not self.visible:
            return
            
        # Draw button background
        color = self.colors['button_pressed'] if self.pressed else self.colors['button_bg']
        pygame.draw.rect(screen, color, self.rect)
        
        # Draw button text
        text_rect = self.font.get_rect(self.text)
        text_x = self.rect.centerx - text_rect.width // 2
        text_y = self.rect.centery - text_rect.height // 2
        self.font.render_to(screen, (text_x, text_y), self.text, self.colors['button_text'])
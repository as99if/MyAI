"""
Base component class for UI elements.
"""

import pygame
import pygame.freetype

class BaseComponent:
    """
    Base class for all UI components.
    """
    
    def __init__(self, rect, font, colors):
        """
        Initialize the base component.
        
        Args:
            rect (pygame.Rect): The rectangle defining the component's position and size.
            font (pygame.freetype.Font): The font to use for text rendering.
            colors (dict): Dictionary of color values for the component.
        """
        self.rect = rect
        self.font = font
        self.colors = colors
        self.visible = True
        self.has_async_event_handler = False
    
    def handle_event(self, event):
        """
        Handle pygame events. Override in subclasses.
        
        Args:
            event (pygame.event.Event): The pygame event to handle.
        """
        pass
    
    def update(self):
        """
        Update component state. Override in subclasses.
        """
        pass
    
    def draw(self, screen):
        """
        Draw the component. Override in subclasses.
        
        Args:
            screen (pygame.Surface): The screen surface to draw on.
        """
        pass
    
    def wrap_text(self, text, width):
        """
        Wrap text to fit within a given pixel width.
        
        Args:
            text (str): The text to wrap.
            width (int): The maximum width in pixels.
        
        Returns:
            list[str]: List of wrapped lines.
        """
        if not text:
            return []
            
        words = text.split(' ')
        lines = []
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            rect = self.font.get_rect(test_line)
            if rect.width > width and line:
                lines.append(line)
                line = word
            else:
                line = test_line
        if line:
            lines.append(line)
        return lines
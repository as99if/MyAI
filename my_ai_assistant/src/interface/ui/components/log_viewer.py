"""
Log viewer component for the UI.
"""

import pygame
from .base_component import BaseComponent

class LogViewer(BaseComponent):
    """
    Scrollable log viewer component.
    """
    
    def __init__(self, rect, scrollbar_rect, font, colors, padding=12):
        """
        Initialize the log viewer component.
        
        Args:
            rect (pygame.Rect): The rectangle defining the component's position and size.
            scrollbar_rect (pygame.Rect): The rectangle for the scrollbar.
            font (pygame.freetype.Font): The font to use for text rendering.
            colors (dict): Dictionary of color values.
            padding (int): Internal padding for text.
        """
        super().__init__(rect, font, colors)
        self.scrollbar_rect = scrollbar_rect
        self.padding = padding
        self.content = ""
        self.scroll_offset = 0
        self.font_size = font.size * 0.6
        self.has_async_event_handler = False
    
    def update_content(self, content):
        """
        Update the log content.
        
        Args:
            content (str): The new log content.
        """
        self.content = content
    
    def get_total_height(self):
        """
        Calculate the total pixel height of all log content.
        
        Returns:
            int: The total height in pixels.
        """
        if not self.content:
            return 0
        lines = self.content.split('\n')
        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(self.wrap_text(line, self.rect.width - 2 * self.padding))
        return len(wrapped_lines) * (self.font_size + 6)
    
    def handle_event(self, event):
        """
        Handle mouse scroll events.
        
        Args:
            event (pygame.event.Event): The pygame event to handle.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            if self.rect.collidepoint(mouse_pos) or self.scrollbar_rect.collidepoint(mouse_pos):
                if event.button == 4:  # scroll up
                    self.scroll_offset = max(0, self.scroll_offset - 40)
                elif event.button == 5:  # scroll down
                    max_scroll = max(0, self.get_total_height() - self.rect.height)
                    self.scroll_offset = min(max_scroll, self.scroll_offset + 40)
    
    def draw(self, screen):
        """
        Draw the log viewer component.
        
        Args:
            screen (pygame.Surface): The screen surface to draw on.
        """
        if not self.visible:
            return
            
        # Draw background
        pygame.draw.rect(screen, self.colors['log_bg'], self.rect)
        
        # Draw log content
        if self.content:
            log_lines = self.content.split('\n')
            wrapped_lines = []
            for line in log_lines:
                wrapped_lines.extend(self.wrap_text(line, self.rect.width - 2 * self.padding))
            
            y = self.rect.top - self.scroll_offset + self.padding
            for line in wrapped_lines:
                if y + self.font_size > self.rect.top and y < self.rect.bottom:
                    self.font.render_to(screen, (self.rect.left + self.padding, y), line, self.colors['log'])
                y += self.font_size + 6
        
        # Draw scrollbar
        log_height = self.get_total_height()
        if log_height > self.rect.height:
            bar_height = max(30, self.rect.height * self.rect.height // log_height)
            bar_top = self.rect.top + (self.scroll_offset * (self.rect.height - bar_height)) // max(1, log_height - self.rect.height)
            pygame.draw.rect(screen, self.colors['scrollbar_bg'], self.scrollbar_rect)
            pygame.draw.rect(screen, self.colors['scrollbar_fg'], (self.scrollbar_rect.left, bar_top, self.scrollbar_rect.width, bar_height))
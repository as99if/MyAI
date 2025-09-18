"""
Conversation display component with inline input for the UI.
"""

import datetime
import time
from typing import Any, List
import pygame
from src.interface.ui.components.base_component import BaseComponent

class ConversationBox(BaseComponent):
    """
    Conversation display component with inline input for the UI.

    Features:
    - Real-time conversation display with User and AI messages
    - Inline text input with visual cursor and blinking animation
    - Scrollable message history with mouse wheel and scrollbar support
    - Text wrapping for long messages to fit component width
    - Auto-scroll to bottom when new messages are added or when typing
    - Advanced text editing capabilities:
    - Character-by-character deletion with backspace
    - Word deletion with Ctrl+Backspace
    - Backspace key repeat when held down
    - Clipboard paste support with Ctrl+V
    - Visual feedback with different colors for User and AI messages
    - Scrollbar indicator when content exceeds visible area
    - Submit messages with Enter key (Shift+Enter for multi-line not implemented)
    - Customizable fonts, colors, and padding
    - Event-driven architecture with callback support for message submission
    """
    
    def __init__(self, rect, scrollbar_rect, font, colors, padding=12):
        """
        Initialize the conversation component.
        
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
        self.messages = []
        self.scroll_offset = 0
        self.font_size = font.size
        
        # Input functionality
        self.current_input = ""
        self.is_inputting = True  # Always ready for input
        self.callback = None
        
        # Cursor properties
        self.cursor_visible = True
        self.cursor_blink_time = 500
        self.last_cursor_blink = time.time() * 1000
        
        # Backspace handling
        self.backspace_held = False
        self.backspace_start_time = 0
        self.last_backspace_time = 0
        self.backspace_repeat_delay = 500
        self.backspace_repeat_rate = 100
        self.has_async_event_handler = True
    
    def set_submit_callback(self, callback):
        """
        Set callback function to call when text is submitted.
        
        Args:
            callback (callable): Function to call with submitted text.
        """
        self.callback = callback
    
    async def _submit_input(self):
        """
        Submit the current input and add it as a user message.
        """
        text = self.current_input.strip()

        if text:
            if self.callback:
                await self.callback(text=text)
        self.current_input = ""
        # Remove the mock reply since the callback will handle the response
        # self.add_message("AI", "This is a sample placeholder response.")
        self.scroll_to_bottom()

    def add_message(self, sender, text):
        """
        Add a message to the conversation.
        
        Args:
            sender (str): The sender of the message ("User" or "AI").
            text (str): The message content.
        """
        self.messages.append((sender, text))
        self.scroll_to_bottom()


    def add_ai_response(self, reply: List[Any]):
        """
        Add an AI response to the conversation.
        
        Args:
            text (str): The AI response content.
        """
        if isinstance(reply, list):
            for segment in reply:
                if type(segment) == str:
                    self.add_message("AI", segment)
                elif hasattr(segment, 'content'):
                    self.add_message("AI", segment.content)
                else:
                    self.add_message("AI", "ERROR: Unknown segment type in AI response.")

                time.sleep(0.1)  
        return
        

    
    def scroll_to_bottom(self):
        """
        Scroll to show the latest message or input line.
        """
        total_height = self.get_total_height()
        visible_height = self.rect.height
        self.scroll_offset = max(0, total_height - visible_height)
    
    def get_total_height(self):
        """
        Calculate the total pixel height of all messages plus input line.
        
        Returns:
            int: The total height in pixels.
        """
        height = 0
        # Calculate height for existing messages
        for sender, text in self.messages:
            lines = self.wrap_text(f"{sender}: {text}", self.rect.width - 2 * self.padding)
            height += len(lines) * (self.font_size + 6) + 8
        
        # Add height for current input line
        if self.is_inputting:
            input_text = f"User: {self.current_input}"
            lines = self.wrap_text(input_text, self.rect.width - 2 * self.padding)
            height += len(lines) * (self.font_size + 6) + 8
        
        return height
    
    def _delete_word(self):
        """
        Delete the previous word from the current cursor position.
        """
        if not self.current_input:
            return
        
        # Find the end of the current word (skip trailing spaces)
        i = len(self.current_input) - 1
        while i >= 0 and self.current_input[i].isspace():
            i -= 1
        
        # Find the beginning of the word
        while i >= 0 and not self.current_input[i].isspace():
            i -= 1
        
        # Delete from after the space (or beginning) to the end
        self.current_input = self.current_input[:i + 1]
    
    def update_cursor_blink(self):
        """
        Update cursor blinking state based on time.
        """
        current_time = time.time() * 1000
        if current_time - self.last_cursor_blink > self.cursor_blink_time:
            self.cursor_visible = not self.cursor_visible
            self.last_cursor_blink = current_time
    
    def update_backspace_repeat(self):
        """
        Handle continuous backspace deletion when key is held down.
        """
        if not self.backspace_held or not self.current_input:
            return
        
        current_time = time.time() * 1000
        time_held = current_time - self.backspace_start_time
        time_since_last = current_time - self.last_backspace_time
        
        # Check if we should repeat the backspace action
        should_repeat = False
        
        if time_held > self.backspace_repeat_delay:
            # After initial delay, check repeat rate
            if time_since_last >= self.backspace_repeat_rate:
                should_repeat = True
        
        if should_repeat:
            # Check if Ctrl is currently held (for word deletion)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                self._delete_word()
            else:
                # Delete single character
                self.current_input = self.current_input[:-1]
            
            self.last_backspace_time = current_time
    
    async def handle_event(self, event):
        """
        Handle mouse scroll and keyboard input events.
        
        Args:
            event (pygame.event.Event): The pygame event to handle.
        """
        # Handle scrolling
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            if self.rect.collidepoint(mouse_pos) or self.scrollbar_rect.collidepoint(mouse_pos):
                if event.button == 4:  # scroll up
                    self.scroll_offset = max(0, self.scroll_offset - 40)
                elif event.button == 5:  # scroll down
                    max_scroll = max(0, self.get_total_height() - self.rect.height)
                    self.scroll_offset = min(max_scroll, self.scroll_offset + 40)
        
        elif event.type == pygame.MOUSEWHEEL:
            # Check if mouse is over the component
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                if event.y > 0:  # Scroll up
                    self.scroll_offset = max(0, self.scroll_offset - 40)
                elif event.y < 0:  # Scroll down
                    max_scroll = max(0, self.get_total_height() - self.rect.height)
                    self.scroll_offset = min(max_scroll, self.scroll_offset + 40)
        
        # Handle keyboard input for text input
        elif event.type == pygame.KEYDOWN and self.is_inputting:
            # Reset cursor visibility when typing
            self.cursor_visible = True
            self.last_cursor_blink = time.time() * 1000
            
            if event.key == pygame.K_RETURN and not event.mod & pygame.KMOD_SHIFT:
                await self._submit_input()
            elif event.key == pygame.K_BACKSPACE:
                current_time = time.time() * 1000
                
                if event.mod & pygame.KMOD_CTRL:
                    # Ctrl+Backspace: delete word immediately
                    self._delete_word()
                else:
                    # Regular backspace: delete single character
                    if self.current_input:
                        self.current_input = self.current_input[:-1]
                
                # Start tracking backspace hold
                self.backspace_held = True
                self.backspace_start_time = current_time
                self.last_backspace_time = current_time
                
            elif event.key == pygame.K_v and event.mod & pygame.KMOD_CTRL:
                try:
                    import pyperclip
                    self.current_input += pyperclip.paste()
                except ImportError:
                    pass
            else:
                if event.unicode:
                    self.current_input += event.unicode
            
            # Auto-scroll to bottom when typing
            self.scroll_to_bottom()
        
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_BACKSPACE:
                # Stop tracking backspace hold
                self.backspace_held = False
    
    def get_cursor_position(self):
        """
        Calculate the cursor position for the input line.
        
        Returns:
            tuple: (x, y) position of the cursor, or None if not visible.
        """
        if not self.is_inputting:
            return None
        
        # Calculate the position of the input line
        y_offset = self.get_total_height() - self.scroll_offset
        input_y = self.rect.top + y_offset - (self.font_size + 6) - 8
        
        # Check if the input line is visible
        if input_y < self.rect.top - self.font_size or input_y > self.rect.bottom:
            return None
        
        # Calculate cursor x position
        input_text = f"User: {self.current_input}"
        text_rect = self.font.get_rect(input_text)
        cursor_x = self.rect.left + self.padding + text_rect.width
        
        return (cursor_x, input_y)
    
    def draw(self, screen):
        """
        Draw the conversation component with inline input.
        
        Args:
            screen (pygame.Surface): The screen surface to draw on.
        """
        if not self.visible:
            return
        
        # Update cursor blinking and backspace repeat
        self.update_cursor_blink()
        self.update_backspace_repeat()
            
        # Draw background
        pygame.draw.rect(screen, self.colors['convo_bg'], self.rect)
        
        # Draw existing messages
        y = self.rect.top - self.scroll_offset + self.padding
        for sender, text in self.messages:
            color = self.colors['user'] if sender == "User" else self.colors['ai']
            lines = self.wrap_text(f"{sender}: {text}", self.rect.width - 2 * self.padding)
            for line in lines:
                if y + self.font_size > self.rect.top and y < self.rect.bottom:
                    self.font.render_to(screen, (self.rect.left + self.padding, y), line, color)
                y += self.font_size + 6
            y += 8
        
        # Draw current input line
        if self.is_inputting:
            input_text = f"User: {self.current_input}"
            lines = self.wrap_text(input_text, self.rect.width - 2 * self.padding)
            
            for line in lines:
                if y + self.font_size > self.rect.top and y < self.rect.bottom:
                    self.font.render_to(screen, (self.rect.left + self.padding, y), line, self.colors['user'])
                y += self.font_size + 6
        
        # Draw blinking cursor
        if self.is_inputting and self.cursor_visible:
            cursor_pos = self.get_cursor_position()
            if cursor_pos:
                cursor_x, cursor_y = cursor_pos
                cursor_height = self.font_size
                pygame.draw.line(screen, self.colors.get('cursor', self.colors['text']), 
                            (cursor_x + 3, cursor_y + cursor_height), (cursor_x + cursor_height, cursor_y + cursor_height), 1)
        # Draw scrollbar
        total_height = self.get_total_height()
        if total_height > self.rect.height:
            bar_height = max(30, self.rect.height * self.rect.height // total_height)
            bar_top = self.rect.top + (self.scroll_offset * (self.rect.height - bar_height)) // max(1, total_height - self.rect.height)
            pygame.draw.rect(screen, self.colors['scrollbar_bg'], self.scrollbar_rect)
            pygame.draw.rect(screen, self.colors['scrollbar_fg'], (self.scrollbar_rect.left, bar_top, self.scrollbar_rect.width, bar_height))
"""
Text input component for the UI.
"""
import time
import pygame
from src.interface.ui.components.base_component import BaseComponent

class TextInputBox(BaseComponent):
    """
    Multi-line text input component with comprehensive editing and navigation features.
    
    Features:
    - Multi-line text input with automatic line wrapping
    - Adjustable height with configurable maximum visible lines
    - Vertical scrolling support (mouse wheel and arrow keys)
    - Blinking cursor with proper positioning
    - Text submission on Enter (Shift+Enter for new line)
    - Backspace handling:
      - Single character deletion
      - Word deletion with Ctrl+Backspace
      - Continuous deletion when key is held down
    - Clipboard support (Ctrl+V for paste)
    - Auto-scroll to cursor position when typing
    - Visual scroll indicator with scrollbar track and thumb
    - Configurable padding, font, colors, and maximum lines
    - Submit callback system for handling user input
    - Focus management and cursor visibility control - always focused by default
    """
    
    def __init__(self, rect, font, colors, max_lines=6, padding=12):
        """
        Initialize the text input component.
        
        Args:
            rect (pygame.Rect): The rectangle defining the component's position and size.
            font (pygame.freetype.Font): The font to use for text rendering.
            colors (dict): Dictionary of color values.
            max_lines (int): Maximum number of visible lines.
            padding (int): Internal padding for text.
        """
        super().__init__(rect, font, colors)
        self.text = ""
        self.lines = [""]
        self.max_lines = max_lines
        self.max_visible_lines = 3  # Maximum visible lines before scrolling
        self.padding = padding
        self.min_height = rect.height
        self.font_size = font.size
        self.callback = None
        self.scroll_offset = 0  # Track vertical scroll position
        
        # Cursor properties
        self.cursor_visible = True
        self.cursor_blink_time = 500  # Blink every 500ms
        self.last_cursor_blink = time.time() * 1000  # Convert to milliseconds
        self.is_focused = True  # Text input is focused by default
        
        # Backspace handling for word deletion
        self.backspace_held = False
        self.backspace_start_time = 0
        self.last_backspace_time = 0
        self.backspace_repeat_delay = 500  # Initial delay before repeating (ms)
        self.backspace_repeat_rate = 100   # Repeat rate after initial delay (ms)
        self.has_async_event_handler = False
        
    def _delete_word(self):
        """
        Delete the previous word from the current cursor position.
        """
        if not self.text:
            return
        
        # Find the end of the current word (skip trailing spaces)
        i = len(self.text) - 1
        while i >= 0 and self.text[i].isspace():
            i -= 1
        
        # Find the beginning of the word
        while i >= 0 and not self.text[i].isspace():
            i -= 1
        
        # Delete from after the space (or beginning) to the end
        self.text = self.text[:i + 1]

    def handle_event(self, event):
        """
        Handle keyboard input events.
        
        Args:
            event (pygame.event.Event): The pygame event to handle.
        """
        if event.type == pygame.KEYDOWN:
            # Reset cursor visibility when typing
            self.cursor_visible = True
            self.last_cursor_blink = time.time() * 1000
            
            if event.key == pygame.K_RETURN and not event.mod & pygame.KMOD_SHIFT:
                self.submit_text()
            elif event.key == pygame.K_BACKSPACE:
                current_time = time.time() * 1000
                
                if event.mod & pygame.KMOD_CTRL:
                    # Ctrl+Backspace: delete word immediately
                    self._delete_word()
                else:
                    # Regular backspace: delete single character
                    if self.text:
                        self.text = self.text[:-1]
                
                # Start tracking backspace hold
                self.backspace_held = True
                self.backspace_start_time = current_time
                self.last_backspace_time = current_time
                
            elif event.key == pygame.K_v and event.mod & pygame.KMOD_CTRL:
                try:
                    import pyperclip
                    self.text += pyperclip.paste()
                except ImportError:
                    pass
            elif event.key == pygame.K_UP:
                # Scroll up
                if self.scroll_offset > 0:
                    self.scroll_offset -= 1
            elif event.key == pygame.K_DOWN:
                # Scroll down
                max_scroll = max(0, len(self.lines) - self.max_visible_lines)
                if self.scroll_offset < max_scroll:
                    self.scroll_offset += 1
            else:
                if event.unicode:
                    self.text += event.unicode
            self.update_lines()
        
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_BACKSPACE:
                # Stop tracking backspace hold
                self.backspace_held = False
        
        # Handle mouse wheel scrolling
        elif event.type == pygame.MOUSEWHEEL:
            # Check if mouse is over the component
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                if event.y > 0:  # Scroll up
                    if self.scroll_offset > 0:
                        self.scroll_offset -= 1
                elif event.y < 0:  # Scroll down
                    max_scroll = max(0, len(self.lines) - self.max_visible_lines)
                    if self.scroll_offset < max_scroll:
                        self.scroll_offset += 1

    def update_backspace_repeat(self):
        """
        Handle continuous backspace deletion when key is held down.
        """
        if not self.backspace_held or not self.text:
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
                self.text = self.text[:-1]
            
            self.last_backspace_time = current_time
            self.update_lines()
        
    def set_submit_callback(self, callback):
        """
        Set callback function to call when text is submitted.
        
        Args:
            callback (callable): Function to call with submitted text.
        """
        self.callback = callback
    
    
    
    def update_cursor_blink(self):
        """
        Update cursor blinking state based on time.
        """
        current_time = time.time() * 1000
        if current_time - self.last_cursor_blink > self.cursor_blink_time:
            self.cursor_visible = not self.cursor_visible
            self.last_cursor_blink = current_time
    
    def get_cursor_position(self):
        """
        Calculate the cursor position in pixels.
        
        Returns:
            tuple: (x, y) position of the cursor.
        """
        if not self.lines:
            return (self.rect.left + self.padding, self.rect.top + self.padding)
        
        # Find which line the cursor is on
        cursor_line_idx = len(self.lines) - 1
        cursor_line = self.lines[cursor_line_idx] if self.lines else ""
        
        # Calculate cursor position within the visible area
        visible_line_idx = cursor_line_idx - self.scroll_offset
        
        # If cursor is not in visible area, don't show it
        if visible_line_idx < 0 or visible_line_idx >= self.max_visible_lines:
            return None
        
        # Calculate x position based on text width
        text_rect = self.font.get_rect(cursor_line)
        cursor_x = self.rect.left + self.padding + text_rect.width
        
        # Calculate y position
        cursor_y = self.rect.top + self.padding + visible_line_idx * (self.font_size + 8)
        
        return (cursor_x, cursor_y)
    
    def wrap_text(self, text, max_width):
        """
        Wrap text to fit within the specified width.
        
        Args:
            text (str): The text to wrap.
            max_width (int): Maximum width in pixels.
            
        Returns:
            list: List of wrapped text lines.
        """
        if not text:
            return [""]
        
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            # Test if adding this word would exceed the width
            test_line = current_line + (" " if current_line else "") + word
            text_rect = self.font.get_rect(test_line)
            
            if text_rect.width <= max_width:
                current_line = test_line
            else:
                # If current line is not empty, start a new line
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Word is too long, split it
                    lines.append(word)
                    current_line = ""
        
        # Add the last line if it's not empty
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [""]

    def update_lines(self):
        """
        Update the wrapped lines and adjust scroll position.
        """
        lines = self.text.split('\n')
        wrapped = []
        for line in lines:
            wrapped.extend(self.wrap_text(line, self.rect.width - 2 * self.padding))
        self.lines = wrapped
        
        # Auto-scroll to show the cursor (end of text) when typing
        if len(self.lines) > self.max_visible_lines:
            self.scroll_offset = max(0, len(self.lines) - self.max_visible_lines)
        else:
            self.scroll_offset = 0
    
    def submit_text(self):
        """
        Submit the current text and clear the input.
        """
        text = self.text.strip()
        if text and self.callback:
            self.callback(text)
        self.text = ""
        self.scroll_offset = 0
        self.update_lines()
    
    def draw(self, screen):
        """
        Draw the text input component with scrolling support and blinking cursor.
        
        Args:
            screen (pygame.Surface): The screen surface to draw on.
        """
        if not self.visible:
            return
        
        # Update cursor blinking
        self.update_cursor_blink()
        
        # Handle continuous backspace deletion
        self.update_backspace_repeat()
            
        # Draw background
        pygame.draw.rect(screen, self.colors['input_bg'], self.rect)
        
        # Calculate visible lines based on scroll offset
        start_line = self.scroll_offset
        end_line = min(start_line + self.max_visible_lines, len(self.lines))
        visible_lines = self.lines[start_line:end_line]
        
        # Draw visible text lines
        for idx, line in enumerate(visible_lines):
            y_pos = self.rect.top + self.padding + idx * (self.font_size + 8)
            self.font.render_to(screen, (self.rect.left + self.padding, y_pos), line, self.colors['text'])
        
        # Draw blinking cursor if focused and visible
        if self.is_focused and self.cursor_visible:
            cursor_pos = self.get_cursor_position()
            if cursor_pos:
                cursor_x, cursor_y = cursor_pos
                # Draw cursor as a horizontal line
                cursor_height = self.font_size * 0.8
                pygame.draw.line(screen, self.colors.get('cursor', self.colors['text']), 
                            (cursor_x, cursor_y + cursor_height), (cursor_x + cursor_height, cursor_y + cursor_height), 1)
        
        # Draw scroll indicator if there's more content
        if len(self.lines) > self.max_visible_lines:
            self._draw_scroll_indicator(screen)
    
    def _draw_scroll_indicator(self, screen):
        """
        Draw a scroll indicator on the right side of the component.
        
        Args:
            screen (pygame.Surface): The screen surface to draw on.
        """
        # Calculate scrollbar dimensions
        scrollbar_width = 4
        scrollbar_height = self.rect.height - 2 * self.padding
        scrollbar_x = self.rect.right - scrollbar_width - 2
        scrollbar_y = self.rect.top + self.padding
        
        # Draw scrollbar track
        track_rect = pygame.Rect(scrollbar_x, scrollbar_y, scrollbar_width, scrollbar_height)
        pygame.draw.rect(screen, self.colors.get('scrollbar_track', (200, 200, 200)), track_rect)
        
        # Calculate thumb position and size
        total_lines = len(self.lines)
        visible_ratio = self.max_visible_lines / total_lines
        thumb_height = max(10, int(scrollbar_height * visible_ratio))
        
        scroll_ratio = self.scroll_offset / (total_lines - self.max_visible_lines)
        thumb_y = scrollbar_y + int((scrollbar_height - thumb_height) * scroll_ratio)
        
        # Draw scrollbar thumb
        thumb_rect = pygame.Rect(scrollbar_x, thumb_y, scrollbar_width, thumb_height)
        pygame.draw.rect(screen, self.colors.get('scrollbar_thumb', (120, 120, 120)), thumb_rect)
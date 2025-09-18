import pygame
from typing import List, Tuple
from threading import Lock

from src.utils.log_manager import LoggingManager

class IntroTerminal:
    def __init__(self, width: int = 600, height: int = 600):
        # Initialize pygame if not already initialized
        if not pygame.get_init():
            pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MyAI Terminal Console")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        
        # Font
        self.font_size = 15
        self.font = pygame.font.Font(None, self.font_size)
        self.line_height = self.font_size + 2
        
        # Console state
        self.messages: List[Tuple[str, str]] = []  # (message, color)
        self.max_lines = (height - 40) // self.line_height
        self.scroll_offset = 0
        self.lock = Lock()
        
        # Clock for frame rate
        self.clock = pygame.time.Clock()
        self.running = True
        self.pygame_initialized = True

        # Subscribe to log manager
        self.log_manager = LoggingManager()
        # Don't subscribe here to avoid circular dependencies
        
    def add_log_message(self, message: str, level: str = "INFO", source: str = ""):
        """Add a log message to the console"""
        with self.lock:
            timestamp = pygame.time.get_ticks() // 1000
            formatted_message = f"[{timestamp}s] [{level}] {source}: {message}"
            color = self.get_color_by_level(level)
            
            # Wrap text to fit within window width
            wrapped_lines = self.wrap_text(formatted_message)
            
            # Add each wrapped line as a separate message
            for line in wrapped_lines:
                self.messages.append((line, color))
            
            # Keep only recent messages
            if len(self.messages) > 1000:
                self.messages = self.messages[-500:]
            
            # Auto-scroll to bottom
            self.scroll_to_bottom()

    def wrap_text(self, text: str) -> List[str]:
        """Wrap text to fit within the console width"""
        max_width = self.width - 20  # Account for margins
        lines = []
        
        # Handle multi-line text (split by existing newlines first)
        text_lines = text.split('\n')
        
        for line in text_lines:
            if not line:
                lines.append("")
                continue
                
            # Calculate how many characters fit in one line
            char_width = self.font.size('M')[0]  # Use 'M' as average character width
            max_chars = max_width // char_width
            
            # If line fits, add it as is
            if len(line) <= max_chars:
                lines.append(line)
            else:
                # Split long lines
                words = line.split(' ')
                current_line = ""
                
                for word in words:
                    # Check if adding this word would exceed the line width
                    test_line = current_line + (" " if current_line else "") + word
                    if len(test_line) <= max_chars:
                        current_line = test_line
                    else:
                        # If current_line is not empty, save it and start new line
                        if current_line:
                            lines.append(current_line)
                            current_line = word
                        else:
                            # Word itself is too long, force break it
                            while len(word) > max_chars:
                                lines.append(word[:max_chars])
                                word = word[max_chars:]
                            current_line = word
                
                # Add the last line if not empty
                if current_line:
                    lines.append(current_line)
        
        return lines
    
    def get_color_by_level(self, level: str) -> Tuple[int, int, int]:
        """Get color based on log level"""
        level_colors = {
            "INFO": self.GREEN,
            "ERROR": self.RED,
            "WARNING": self.YELLOW,
            "DEBUG": self.BLUE,
            "CRITICAL": self.RED
        }
        return level_colors.get(level.upper(), self.WHITE)
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the console"""
        if len(self.messages) > self.max_lines:
            self.scroll_offset = len(self.messages) - self.max_lines
        else:
            self.scroll_offset = 0
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
        
        return True
    
    def draw(self):
        """Draw the console"""
        self.screen.fill(self.BLACK)
        
        # Draw messages
        y_offset = 10
        visible_messages = self.messages[self.scroll_offset:self.scroll_offset + self.max_lines]
        
        for message, color in visible_messages:
            text_surface = self.font.render(message, True, color)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += self.line_height
        
        # Draw status bar
        status_text = f"Messages: {len(self.messages)} | ESC to exit"
        status_surface = self.font.render(status_text, True, self.GRAY)
        self.screen.blit(status_surface, (10, self.height - 20))
        
        pygame.display.flip()
    
    def run_frame(self):
        """Run a single frame of the console"""
        if not self.handle_events():
            return False
        
        self.draw()
        self.clock.tick(60)  # 60 FPS
        return True
    
    def close(self):
        """Close the console"""
        self.running = False
        if self.pygame_initialized and pygame.get_init():
            try:
                pygame.display.quit()
            except:
                pass
"""
MyAIUI

This module implements a simple, terminal-style chatbot UI using pygame.
Features:
- Dark terminal-like theme
- Adjustable font size and font
- buttons and rectangles are not rounded
- Three main areas: scrollable conversation display and multi-line text input on the left column,  a log viewer on the right column 
- Input can be submitted with Enter key or Send button
- Conversation area is scrollable with mouse wheel
- Messages can be appended programmatically
- log viewer will get updated from log_manager

Classes:
    MyAIUI: Main class for the chatbot UI.
"""

import asyncio
import datetime
import pprint
import pygame
import pygame.freetype
import sys
import os

from src.core.api_server.data_models import ContentSegment, MessageContent
from src.core.my_ai_assistant import MyAIAssistant
from src.interface.ui.components.buttons import SendButton
from src.interface.ui.components.conversation_box import ConversationBox
from src.interface.ui.components.log_viewer import LogViewer
from src.interface.ui.components.audio_visualizer import AudioVisualizer
from src.utils.log_manager import AgentLoggingManager, LoggingManager

# Add the utils directory to the path to import log_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))


# --- Configurable Variables ---
FONT_SIZE = 15  # Adjust this to change font size
FONT_NAME = "DejaVuSansMono"  # Linux terminal-like font
MAX_INPUT_LINES = 6

# --- Colors for Dark Terminal Theme ---
COLORS = {
    'bg': (30, 32, 34),
    'input_bg': (40, 42, 46),
    'convo_bg': (24, 26, 28),
    'log_bg': (28, 30, 32),
    'text': (204, 204, 204),
    'user': (80, 200, 120),
    'ai': (120, 180, 255),
    'log': (180, 180, 120),
    'button_bg': (60, 63, 65),
    'button_pressed': (80, 83, 85),
    'button_text': (220, 220, 220),
    'scrollbar_bg': (50, 52, 54),
    'scrollbar_fg': (90, 92, 94)
}

WIDTH, HEIGHT = 1200, 800
INPUT_HEIGHT = 90
PADDING = 6
BUTTON_WIDTH = 40
BUTTON_HEIGHT = 40
SCROLLBAR_WIDTH = 12
LOG_PANEL_WIDTH = 400  # Width of the log panel on the right

class MyAIUI:
    """
    Implements a terminal-style chatbot UI using pygame.
    """

    def __init__(self, my_ai_assistant:MyAIAssistant=None):
        """
        Initialize the UI, pygame, fonts, rectangles, and state variables.
        """
        # pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("MyAI Terminal Chat")
        self.font = pygame.freetype.SysFont(FONT_NAME, FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Initialize logging managers
        self.log_manager = LoggingManager()
        self.agent_log_manager = AgentLoggingManager()
        
        # Setup layout
        self.setup_layout()
        
        # Subscribe to log updates
        self.log_manager.subscribe(self.update_log_content)
        self.agent_log_manager.subscribe(self.update_log_content)
        
        self.my_ai_assistant = my_ai_assistant
        if self.my_ai_assistant is None:
            self.log_manager.add_message("MyAIAssistant not connected", "INFO", "MyAIUI")
        else:
            # Setup TTS callbacks for audio visualization
            if hasattr(self.my_ai_assistant.speech_engine, 'tts_engine'):
                tts_engine = self.my_ai_assistant.speech_engine.tts_engine
                tts_engine.set_audio_data_callback(self.audio_visualizer.set_audio_data)
                tts_engine.set_playback_position_callback(self.audio_visualizer.set_playback_state)
                tts_engine.set_text_stream_callback(self._handle_tts_text_stream)
    
        self.conversation.add_ai_response([f"------------------------------"])

    def setup_layout(self):
        """
        Setup the component layout and initialize all UI components.
        """
        # Adjust layout to include audio visualizer
        left_column_width = WIDTH - LOG_PANEL_WIDTH - 3 * PADDING
        
        # Split left column into conversation (top) and audio visualizer (bottom)
        visualizer_height = 200
        conversation_height = HEIGHT - visualizer_height - 3 * PADDING
        
        # Left column rectangles
        conversation_rect = pygame.Rect(PADDING, PADDING, 
                                       left_column_width - SCROLLBAR_WIDTH, 
                                       conversation_height)
        conversation_scrollbar_rect = pygame.Rect(left_column_width - SCROLLBAR_WIDTH, PADDING, 
                                                 SCROLLBAR_WIDTH, conversation_height)
        
        # Audio visualizer rectangle
        visualizer_rect = pygame.Rect(PADDING, PADDING + conversation_height + PADDING,
                                     left_column_width, visualizer_height)
        
        # Right column rectangles (log panel)
        log_x = left_column_width
        
        log_rect = pygame.Rect(log_x, PADDING, LOG_PANEL_WIDTH - SCROLLBAR_WIDTH - PADDING, 
                              HEIGHT - 2 * PADDING)
        
        # TODO: add a component to visualize audio fr levels with colors
        log_scrollbar_rect = pygame.Rect(log_x + LOG_PANEL_WIDTH - SCROLLBAR_WIDTH - PADDING, PADDING, 
                                        SCROLLBAR_WIDTH, HEIGHT - 2 * PADDING)
        
        # Initialize components
        # self.text_input = UserMessageInput(input_rect, self.font, COLORS, MAX_INPUT_LINES, PADDING)
        # self.text_input.set_submit_callback(self.handle_user_input)
        
        self.conversation = ConversationBox(conversation_rect, conversation_scrollbar_rect, self.font, COLORS, PADDING)
        self.conversation.set_submit_callback(self.handle_user_input)
        self.audio_visualizer = AudioVisualizer(visualizer_rect, self.font, COLORS, PADDING)
        self.log_viewer = LogViewer(log_rect, log_scrollbar_rect, self.font, COLORS, PADDING)
        
        # self.send_button = SendButton(button_rect, self.font, COLORS)
        # self.send_button.set_click_callback(self.text_input.submit_text)
        
        # Store components for easy iteration
        # self.components = [self.text_input, self.conversation, self.log_viewer, self.send_button]
        self.components = [self.conversation, self.log_viewer, self.audio_visualizer]

    def _handle_tts_text_stream(self, text_chunk: str):
        """
        Handle streaming text updates from TTS engine.
        
        Args:
            text_chunk (str): New text chunk from TTS.
        """
        # This can be used to update the conversation in real-time
        # For now, we'll just pass it to the audio visualizer
        pass

    async def handle_user_input(self, text):
        """
        Handle user text content from ConversationBox or anywhere and pass tho MyAIAssistant.
        
        Args:
            text (str): The user's input text.
        """
        
        if self.my_ai_assistant is None:
            self.log_manager.add_message("MyAIAssistant not connected", "INFO", "MyAIUI")
            self.conversation.add_ai_response(["MyAIAssistant not connected"])
            return
        else:
            message = self.my_ai_assistant.process_input("User", text=text)
            # pprint.pprint(message)
            self.conversation.add_message("User", text)
            responses, recent_conversations = await self.my_ai_assistant.process_and_create_chat_generation(
                message=message
            )
            # pprint.pprint(f"*** --- Responses conversations:\n{pprint.pformat(responses)}")

            # pprint.pprint(f"*** --- Recent conversations:\n{pprint.pformat(recent_conversations)}")
        
            # TODO: reply display will be handled from the TTS engine callback in the future
            # if not SpeechEngine
            # for response in responses:
            self.conversation.add_ai_response(responses)  # Display first response for now
        
        # self.append_message("AI","another mock reply from AI") 
        
        return

    def update_log_content(self, logs):
        """
        Update the log content when logs change.
        
        Args:
            logs (str): The updated log content.
        """
        # Combine both system and agent logs
        system_logs = self.log_manager.get_logs()
        agent_logs = self.agent_log_manager.get_logs()
        
        if system_logs and agent_logs:
            combined_logs = system_logs + "\n" + agent_logs
        elif system_logs:
            combined_logs = system_logs
        elif agent_logs:
            combined_logs = agent_logs
        else:
            combined_logs = ""
        
        self.log_viewer.update_content(combined_logs)

    def append_message(self, sender, text):
        """
        Append a message to the conversation area.

        Args:
            sender (str): The sender of the message ("User" or "AI").
            text (str): The message content.
        """
        self.conversation.add_message(sender, text)


    def draw(self):
        """
        Render the entire UI.
        """
        self.screen.fill(COLORS['bg'])
        
        # Draw all components
        for component in self.components:
            component.draw(self.screen)
        
        pygame.display.flip()
    
    def graceful_exit(self):
        """
        Handle graceful shutdown of the UI and cleanup resources.
        """
        self.log_manager.add_message("UI shutting down", "INFO", "MyAIUI")
        
        # Unsubscribe from log updates
        try:
            self.my_ai_assistant.exit_gracefully()
        except Exception as e:
            self.log_manager.add_message("Error in usnsubscribing from MyAIAssistant", "INFO", "MyAIUI")
            
        try:
            self.log_manager.unsubscribe(self.update_log_content)
            self.agent_log_manager.unsubscribe(self.update_log_content)
        except Exception as e:
            print(f"Error unsubscribing from logs: {e}")
            self.log_manager.add_message("Error in unsubscribing from logs", "INFO", "MyAIUI")
        
        # Set running to False to exit main loop
        self.running = False
        
        # Quit pygame
        pygame.quit()

    async def _run(self):
        """
        Main loop for the UI. Handles events and updates the display.
        """
        # Add some initial log messages for testing
        self.log_manager.add_message("UI initialized", "INFO", "MyAIUI")
        self.agent_log_manager.add_message("test log", "INFO", "MyAIUI")
        
        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.graceful_exit()
                        return
                    else:
                        # Pass events to all components
                        for component in self.components:
                            if component.has_async_event_handler:
                                await component.handle_event(event)
                            else:
                                component.handle_event(event)
                
                # Update all components
                for component in self.components:
                    component.update()
                
                # Draw everything
                self.draw()
                self.clock.tick(60)
        except KeyboardInterrupt:
            self.log_manager.add_message("Received keyboard interrupt", "INFO", "MyAIUI")
            self.graceful_exit()
        except Exception as e:
            self.log_manager.add_message(f"Unexpected error: {e}", "ERROR", "MyAIUI")
            self.graceful_exit()
            raise

    def run(self):
        """
        Start the UI main loop.
        """
        asyncio.run(self._run())

# Example usage
if __name__ == "__main__":
    ui = MyAIUI()
    
    ui.run()
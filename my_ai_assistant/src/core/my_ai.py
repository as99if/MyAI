import asyncio
import threading
import time
import pygame

from src.config.config import load_config

from src.memory_processor.memory_processor import MemoryProcessor
# from src.core.api_server.api import MyAIChatAPI
from src.core.my_ai_assistant import MyAIAssistant
from src.inference_engine.inference_processor import InferenceProcessor
from src.interface.speech_engine import SpeechEngine
from src.utils.log_manager import LoggingManager

# import uvicorn

class MyAI:
    def __init__(self, is_gui_enabled=True):
        self.logging_manager = LoggingManager()
        self.is_gui_enabled = is_gui_enabled
        self.is_loading: bool = False
        self.my_ai_assistant = None
        self.config = None
        self.app_config = None
        self.speech_engine = None
        self.conveversation_history_engine = None
        self.inference_processor = None
        self.terminal_console = None
        self.console_thread = None

    
    def start_loading_terminal(self):
        """Start the terminal console in a separate thread"""
        from src.core.intro_terminal import IntroTerminal
        # Initialize pygame for the intro UI
        pygame.init()
        print("-------------- Starting My AI --------------")
        self.terminal_console = IntroTerminal()
        
        # Redirect logging to console
        self.logging_manager.subscribe(self.terminal_console.add_log_message)
    
    def stop_loading_terminal(self):
        """Safely stop the terminal console"""
        if self.terminal_console:
            self.terminal_console.close()
            # Give it a moment to close properly
            time.sleep(0.5)
            self.terminal_console = None

    async def run(self):
        # Start terminal console first
        self.start_loading_terminal()
        time.sleep(1)  # Give some time for the terminal to initialize
        
        try:
            # Add initial message
            self.logging_manager.add_message("System starting...", level="INFO", source="MyAI")
            
            # Give the terminal time to display the initial message
            for _ in range(30):  # Run a few frames to display the initial message
                if not self.terminal_console or not self.terminal_console.running:
                    break
                if not self.terminal_console.run_frame():
                    break
                await asyncio.sleep(0.016)
            
            # Keep the terminal running while we load components
            loading_complete = False
            components_loaded = False
            
            while self.terminal_console and self.terminal_console.running and not loading_complete:
                # Run one frame of the terminal
                if not self.terminal_console.run_frame():
                    break
                    
                # Only load components once
                if not components_loaded:
                    self.is_loading = True
                    try:
                        print('MyAI Initiating ...')
                        self.logging_manager.add_message("Initiating MyAI...", level="INFO", source="MyAI")
                        
                        # Run terminal frames to display the message
                        for _ in range(30):
                            if not self.terminal_console or not self.terminal_console.running:
                                break
                            if not self.terminal_console.run_frame():
                                break
                            await asyncio.sleep(0.016)
                        
                        self.config = load_config("src/config/config.json")
                        self.logging_manager.add_message("Configuration loaded", level="INFO", source="MyAI")
                        
                        # Run terminal frames to display the message
                        for _ in range(30):
                            if not self.terminal_console or not self.terminal_console.running:
                                break
                            if not self.terminal_console.run_frame():
                                break
                            await asyncio.sleep(0.016)
                        
                        self.speech_engine = SpeechEngine(debug=True)
                        self.logging_manager.add_message("Speech engine loaded successfully", level="INFO", source="MyAI")
                        
                        # Run terminal frames to display the message
                        for _ in range(30):
                            if not self.terminal_console or not self.terminal_console.running:
                                break
                            if not self.terminal_console.run_frame():
                                break
                            await asyncio.sleep(0.016)
                        
                        self.memory_processor = MemoryProcessor()
                        await self.memory_processor.connect()
                        self.logging_manager.add_message("Connected to memory services", level="INFO", source="MyAI")
                        
                        # Run terminal frames to display the message
                        for _ in range(30):
                            if not self.terminal_console or not self.terminal_console.running:
                                break
                            if not self.terminal_console.run_frame():
                                break
                            await asyncio.sleep(0.016)
                        
                        self.inference_processor = InferenceProcessor()
                        self.logging_manager.add_message("Inference engine initiated", level="INFO", source="MyAI")
                        
                        # Run terminal frames to display the message
                        for _ in range(30):
                            if not self.terminal_console or not self.terminal_console.running:
                                break
                            if not self.terminal_console.run_frame():
                                break
                            await asyncio.sleep(0.016)

                        try:
                            self.my_ai_assistant = MyAIAssistant(
                                    inference_processor=self.inference_processor,
                                    memory_processor=self.memory_processor,
                                    speech_engine=self.speech_engine
                                )
                            self.logging_manager.add_message("MyAI Assistant initialized successfully", level="INFO", source="MyAI")
                        except Exception as e:
                            self.logging_manager.add_message(f"Error initializing MyAIAssistant: {str(e)}", level="ERROR", source="MyAI")
                            raise e
                        
                        # Run terminal frames to display the final messages
                        for _ in range(30):
                            if not self.terminal_console or not self.terminal_console.running:
                                break
                            if not self.terminal_console.run_frame():
                                break
                            await asyncio.sleep(0.016)

                        components_loaded = True
                        self.logging_manager.add_message("Initializing UI...", level="INFO", source="MyAI")
                        
                        # Run terminal frames to display the final message
                        for _ in range(60):  # Give more time for final message
                            if not self.terminal_console or not self.terminal_console.running:
                                break
                            if not self.terminal_console.run_frame():
                                break
                            await asyncio.sleep(0.016)
                        
                        loading_complete = True
                        
                    except Exception as e:
                        self.is_loading = False
                        self.logging_manager.add_message(f"Error during initialization: {str(e)}", level="ERROR", source="MyAI")
                        # Run terminal frames to display error
                        for _ in range(120):  # Show error for longer
                            if not self.terminal_console or not self.terminal_console.running:
                                break
                            if not self.terminal_console.run_frame():
                                break
                            await asyncio.sleep(0.016)
                        raise Exception(f"Error during initialization: {str(e)}")
                
                # Small delay between frames
                await asyncio.sleep(0.016)
            
            # Stop the terminal before starting the UI
            self.stop_loading_terminal()
            
            # Small delay to ensure pygame is properly cleaned up
            time.sleep(0.5)
            
            # Initialize pygame again for the UI
            pygame.init()
            
            # init MyAIUI here
            from src.interface.ui.my_ai_ui import MyAIUI
            self.my_ai_ui = MyAIUI(my_ai_assistant=self.my_ai_assistant)
            
            # Add debugging before running UI
            try:
                await self.my_ai_ui._run()
            except Exception as ui_error:
                print(f"UI Error: {str(ui_error)}")
                raise ui_error
            
            self.is_loading = False
            
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.stop_loading_terminal()
        
    def __run__(self):
        asyncio.run(self.run())
import pprint
from typing import List
import gradio as gr
import random
import time
import logging
from src.utils.log_manager import LoggingManager
from src.core.api_server.data_models import MessageContent
from src.core.my_ai_assistant import MyAIAssistant
from src.inference_engine.inference_processor import InferenceProcessor

class MyAIUI:
    
    def __init__(self, inference_processor: InferenceProcessor = None, my_ai_assistant: MyAIAssistant = None):
        
        self.inference_processor = inference_processor
        self.my_ai_assistant = my_ai_assistant
        self.messages = None
        # Sample responses - replace with your actual chatbot logic
        self.sample_responses = [
            "Hello! How can I help you today?",
            "That's an interesting question. Let me think about that.",
            "I'm a simple chatbot. My responses are pre-programmed.",
            "Could you please elaborate on that?",
            "Thanks for sharing! Is there anything else you'd like to discuss?",
            "I'm here to assist you with any questions you might have."
        ]
        self.css = """
        .log_display {
            font-size: 0.4em;
            font-family: monospace;
            line-height: 1.2;
            white-space: pre-wrap;
            color: #4a4a4a;
        }
        """
        
        # Initialize logging manager
        self.logging_manager = LoggingManager()
        self.logging_manager.subscribe(self.update_log_display)
        
        
        # Create a custom theme
        self.theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        ).set()
        self.is_my_ai_inistialized: bool = False
        self.is_loading: bool = False
        self.app_ui = None
        self._initialize_my_ai()
        if self.is_my_ai_inistialized:
            self._initialize_app_ui()
        

    
    def _initialize_my_ai(self):
        self.inference_processor = InferenceProcessor()
        self.my_ai_assistant = MyAIAssistant(
            inference_processor=self.inference_processor,
        )
        self.is_my_ai_inistialized = True
    
    def update_log_display(self, logs: str):
        """Callback to update the UI log display"""
        if hasattr(self, 'log_display'):
            self.log_display.value=logs
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add a log message to the logging panel"""
        self.logging_manager.add_message(message, level, source="UI")
        return self.logging_manager.get_logs()
        
    async def send_message(self, message: str, history: List[str] = []):
        # Add a small delay to simulate the assistant "thinking" 
        self.messages.value=[history, message]
        time.sleep(0.5)
        
        # Get random response from samples
        # bot_response = random.choice(self.sample_responses)
        
        _message: MessageContent = MessageContent(
            role="user",
            content=message,
            timestamp=time.time(),
        )
        _response, recent_messages = await self.my_ai_assistant.process_and_create_chat_generation(
            message=_message,
            is_tool_call_permitted=True
        )
        pprint.pprint(_response)
        response = _response.content
        
            
        # For chatbot, we need to return a list of message pairs
        history = history
        history.append([message, response])
        return history

    def _initialize_app_ui(self):
        with gr.Blocks(theme=self.theme, css=self.css) as self.app_ui:
            gr.HTML("<div class='title'>Simple Chatbot Assistant</div>")
            
            with gr.Row():
                # Chat column
                with gr.Column(scale=2):
                    self.messages = gr.Chatbot(
                        value=[["Yo", "Yo."]],
                        elem_classes=["chatbot"],
                        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=Assistant"),
                        height=500,
                        show_label=False,
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            container=False,
                            scale=9,
                            show_label=False,
                            autofocus=True
                        )
                        submit_btn = gr.Button(
                            "Send",
                            scale=1,
                            variant="primary",
                            size="lg"
                        )
                
                # Logging column
                with gr.Column(scale=1):
                    self.log_display = gr.Textbox(
                        label="System Logs",
                        value="",
                        lines=25,
                        max_lines=25,
                        interactive=False,
                        elem_classes=["log_display"],
                    )
            
            # Event handlers
            msg.submit(
                fn=self.send_message,
                inputs=[msg, self.messages],
                outputs=[self.messages],
                queue=False
            ).then(
                fn=lambda: "",  # Clear input box after sending
                outputs=[msg]
            ).then(
                fn=self.logging_manager.get_logs,  # Update logs after message
                outputs=[self.log_display]
            )
            
            submit_btn.click(
                fn=self.send_message,
                inputs=[msg, self.messages],
                outputs=[self.messages],
                queue=False
            ).then(
                fn=lambda: "",  # Clear input box after sending
                outputs=[msg]
            ).then(
                fn=self.logging_manager.get_logs,  # Update logs after message
                outputs=[self.log_display]
            )

if __name__ == "__main__":
    app = MyAIUI()
    app.app_ui.launch(debug=True)
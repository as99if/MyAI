import asyncio
from datetime import datetime
import json
import pprint
import gradio as gr
from typing import List, Dict, Any, Optional, Tuple
from src.utils.my_ai_utils import format_messages
from src.memory_processor.memory_processor import MemoryProcessor
from src.core.api_server.data_models import ContentSegment, MessageContent
from src.inference_engine.inference_processor import InferenceProcessor
from src.utils.log_manager import LoggingManager
from src.core.my_ai_assistant import MyAIAssistant
from src.interface.ui.ui_utils import get_custom_css
from src.interface.ui.styling import js_func
from gradio import ChatMessage
from gradio.themes.utils import sizes

class MyAIInterface:
    def __init__(self, memory_processor: Optional[MemoryProcessor] = None, inference_processor: Optional[InferenceProcessor] = None, my_ai_assistant: Optional[MyAIAssistant] = None):
        self.logging_manager = LoggingManager()
        self.logging_manager.subscribe(self.update_log_display)

        self.enable_agent = False
        self.memory_processor: MemoryProcessor = memory_processor
        self.inference_processor: InferenceProcessor = inference_processor
        self.my_ai_assistant: MyAIAssistant = my_ai_assistant
        self.interface = self.build_interface()

        self._initialize_my_ai()
    
    def _initialize_my_ai(self):
        """Initialize the AI components if not provided during instantiation"""
        try:
            if not self.memory_processor:
                self.memory_processor = MemoryProcessor()
                asyncio.run(self.memory_processor.connect())

            if not self.inference_processor:
                self.inference_processor = InferenceProcessor()
            
            if not self.my_ai_assistant:
                self.my_ai_assistant = MyAIAssistant(
                    memory_processor=self.memory_processor,
                    inference_processor=self.inference_processor,
                )
            
            self.is_my_ai_initialized = True
            self.log_message("MyAIUI initialized successfully", "INFO")
        except Exception as e:
            self.log_message(f"Failed to initialize MyAIUI: {str(e)}", "ERROR")
            raise
    
    def log_message(self, message: str, level: str = "INFO") -> str:
        """Add a log message to the logging panel"""
        self.logging_manager.add_message(message, level, source="MY_AI_UI")
        return self.logging_manager.get_logs()

    def update_log_display(self, logs: str):
        if hasattr(self, 'log_display'):
           self.log_display.value = logs
        

    def toggle_agent(self, value=None):
        """Toggle or set the agent enabled state"""
        if value is not None:
            self.enable_agent = value
        else:
            self.enable_agent = not self.enable_agent
        
        status = 'enabled' if self.enable_agent else 'disabled'
        self.log_message(f"Agent {status}", "INFO")
        return self.enable_agent
    

    async def get_chat_completion(self, message: MessageContent, if_agent:bool, nb_retries:int=3, delay:int=30) -> Tuple[MessageContent, List[MessageContent]]:
        """
        Sends a request to the ChatGPT API to retrieve a response based on a list of previous messages.
        """
        response = None
        recent_conversation = None
        try:
            # print("*****MESSAGE****")
            # pprint.pprint(message)
            response, recent_conversation = await self.my_ai_assistant.process_and_create_chat_generation(
                message=message,
                is_tool_call_permitted=if_agent
            )
            # print("*****RESPONSE****")
            # pprint.pprint(response)
            
            return response, recent_conversation
        except Exception as e:
            self.log_message(f"Error getting response: {e}")
            raise
    
    def format_conversation_history(self, messages: List[MessageContent] = None)-> List[ChatMessage]:
        """Removing the unspoken messages and system level prompts from conversation view"""
        # print("*******before formatting for ui******")  
        # pprint.pprint(messages)
        _messages = []
        for msg in messages:
            if not msg.unspoken_message:
                _messages.append(msg)
        # print("\n\n*******after formatting for ui******")
        # pprint.pprint(_messages)
        return _messages

    def get_reply(self, message, history, if_agent:bool = False):
        """
        Predict the response of the chatbot and complete a running list of chat history.
        """
        self.log_message(f"Sending user message: {message}")
        
        # Create message content object
        _message = MessageContent(
            role="user",
            content=[ContentSegment(type="text", text=message)],
            timestamp=datetime.now()
        )
        

        _response, _recent_conversations = asyncio.run(self.get_chat_completion(_message, if_agent=if_agent))

        self.log_message(f"Got response: {_response.content}")
        # print("*******---response------******")
        # pprint.pprint(_response)
        # print("*******---------******\n\n")

        # print("*******---recent_conversations------******")
        # pprint.pprint(_recent_conversations)
        # print("*******---------******\n\n")

        # history.append(_response)
        history = self.format_conversation_history(_recent_conversations)
        history = format_messages(history)

        # history.append(ChatMessage(role=_response.role, content=_response.content))
        return history

    def build_interface(self):
        with gr.Blocks(
            theme=gr.themes.Monochrome(text_size=sizes.text_sm),
            css=get_custom_css(),
            js=js_func,
            fill_height=True,
            fill_width=True,
            elem_id="main_container"     
        ) as interface:
            gr.Markdown("# MyAI Assistant")
            
            with gr.Row(elem_id="main_content"):
                with gr.Column(scale=2, elem_id="media_column"):
                    with gr.Row(elem_id="video_container"):
                        # square panel for video feed
                        self.video_display = gr.TextArea(
                            label="",
                            lines=10,
                            interactive=False,
                            show_copy_button=True,
                            container=False,
                            elem_id="video_display"
                        )
                    with gr.Row(elem_id="audio_container"):
                        # square panel for audio visualizer
                        self.audio_display = gr.TextArea(
                            label="",
                            lines=10,
                            interactive=False,
                            show_copy_button=True,
                            container=False,
                            elem_id="audio_display"
                        )
                with gr.Column(scale=3, show_progress=True, elem_id="chat_column"):
                    # Initialize chatbot component
                    self.chatbot = gr.Chatbot(
                        height="600px",
                        show_label=False,
                        container=True,
                        type="messages",
                        elem_id="chatbot_panel"
                    )
                with gr.Column(scale=2, elem_id="info_column"):
                    
                    with gr.Row(show_progress=True, elem_id="info_display_container"):
                        self.something_display1 = gr.TextArea(
                            label="",
                            lines=10,
                            interactive=False,
                            show_copy_button=True,
                            container=False,
                            elem_id="info_display"
                        )
                    # Log viewer
                    with gr.Row(
                        show_progress=True,
                        elem_id="logs_container",
                    ):
                        self.log_display = gr.TextArea(
                            label="System Logs",
                            lines=2,
                            interactive=False,
                            show_copy_button=True,
                            container=False,
                            elem_id="logs_panel"
                        )
                    
            with gr.Row(elem_id="input_container"):
                # Message input and send button
                self.msg_input = gr.Textbox(
                        label="User Message",
                        placeholder="Type your message here...",
                        container=False,
                        stop_btn=True,
                        show_copy_button=False,
                        lines=3,
                        submit_btn=True,
                        show_label=False,
                        elem_id="user_message"
                    )
            with gr.Row(elem_id="footer_elems"):
                with gr.Column(scale=3, show_progress=True):
                    self.something_display2 = gr.TextArea(
                            label="",
                            lines=2,
                            interactive=False,
                            show_copy_button=True,
                            container=False,
                            elem_id="footer_display"
                        )
                with gr.Column(scale=1):
                    self.agent_btn = gr.Button(
                            "Ask MyAI Agent",
                            size="md",
                            scale=1,
                            variant="stop",
                            elem_id="agent_button",
                            elem_classes="inference_button"
                        )
            
            # Setup event handlers
            self._setup_event_handlers()
        
        interface.queue()
        return interface
    
    def _setup_event_handlers(self):
        """Set up all event handlers for the UI"""
        # Send message handler - triggered by both button and Enter key

        self.msg_input.submit(
            fn=self.get_reply,
            inputs=[self.msg_input, self.chatbot],
            outputs=[self.chatbot],
            api_name=False
        ).then(
            fn=lambda: "",
            outputs=[self.msg_input]
        ).then(
            fn=self.logging_manager.get_logs,
            outputs=[self.log_display]
        )
        
        # Agent button handler
        self.agent_btn.click(
            fn=lambda msg, hist: self.get_reply(msg, hist, True),
            inputs=[self.msg_input, self.chatbot],
            outputs=[self.chatbot],
            api_name="send_message"
        ).then(
            fn=lambda: "",
            outputs=[self.msg_input]
        ).then(
            fn=self.logging_manager.get_logs,
            outputs=[self.log_display]
        )

if __name__ == "__main__":
    chat_interface = MyAIInterface()
    chat_interface.interface.launch(share=False, pwa=True)
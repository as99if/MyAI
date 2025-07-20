import sys
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QSizePolicy, QSplitter
)
from PyQt6.QtCore import Qt
import asyncio
from src.interface.speech_visualizer import AudioVisualizer
from src.interface.speech_engine import SpeechEngine
from src.utils.my_ai_utils import format_messages
from src.memory_processor.memory_processor import MemoryProcessor
from src.core.api_server.data_models import ContentSegment, MessageContent
from src.inference_engine.inference_processor import InferenceProcessor
from src.utils.log_manager import LoggingManager
from src.core.my_ai_assistant import MyAIAssistant

class MyAIInterface(QWidget):
    def __init__(self, memory_processor=None, inference_processor=None, my_ai_assistant=None):
        super().__init__()
        self.setWindowTitle("MyAI Assistant")
        self.resize(1400, 800)

        # Backend
        self.logging_manager = LoggingManager()

        self.enable_agent = False
        self.memory_processor = memory_processor or MemoryProcessor()
        self.inference_processor = inference_processor or InferenceProcessor()
        self.audio_display = AudioVisualizer(width=600, height=200)
        self.speech_engine = SpeechEngine()  # Placeholder for speech engine
        
        self.my_ai_assistant = my_ai_assistant or MyAIAssistant(
            memory_processor=self.memory_processor,
            inference_processor=self.inference_processor,
            speech_engine=self.speech_engine
        )
        self.my_ai_assistant.tts_engine.set_visualizer(self.audio_display)

        # Layouts
        main_layout = QVBoxLayout(self)
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Media Column ---
        media_widget = QWidget()
        media_layout = QVBoxLayout(media_widget)
        self.video_display = QTextEdit()
        self.video_display.setReadOnly(True)
        self.video_display.setPlaceholderText("Video feed placeholder")
        # self.audio_display = QTextEdit()
        # self.audio_display.setReadOnly(True)
        # self.audio_display.setPlaceholderText("Audio visualizer placeholder")
        media_layout.addWidget(QLabel("Video"))
        media_layout.addWidget(self.video_display)
        media_layout.addWidget(QLabel("Audio"))
        media_layout.addWidget(self.audio_display)
        content_splitter.addWidget(media_widget)

        # --- Chat Column ---
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        self.chat_list = QListWidget()
        self.chat_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        chat_layout.addWidget(QLabel("Chatbot"))
        chat_layout.addWidget(self.chat_list)
        content_splitter.addWidget(chat_widget)

        # --- Info/Log Column ---
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setPlaceholderText("Info panel placeholder")
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(60)
        # Subscribe to logs AFTER UI is ready
        self.logging_manager.subscribe(callback=self.update_log_display)
        
        info_layout.addWidget(QLabel("Info"))
        info_layout.addWidget(self.info_display)
        info_layout.addWidget(QLabel("System Logs"))
        info_layout.addWidget(self.log_display)
        content_splitter.addWidget(info_widget)

        main_layout.addWidget(content_splitter)

        # --- Input Row ---
        input_layout = QHBoxLayout()
        self.msg_input = QLineEdit()
        self.msg_input.setPlaceholderText("Type your message here...")
        self.msg_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.msg_input)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)
        self.agent_btn = QPushButton("Ask MyAI Agent")
        self.agent_btn.clicked.connect(self.send_agent_message)
        input_layout.addWidget(self.agent_btn)
        main_layout.addLayout(input_layout)

        # --- Footer Row ---
        footer_layout = QHBoxLayout()
        self.footer_display = QTextEdit()
        self.footer_display.setReadOnly(True)
        self.footer_display.setMaximumHeight(40)
        self.footer_display.setPlaceholderText("Footer info here")
        footer_layout.addWidget(self.footer_display)
        main_layout.addLayout(footer_layout)

        # State
        self.chat_history = []
        self.log_message("MyAIUI initialized successfully", "INFO")

    def log_message(self, message, level="INFO"):
        self.logging_manager.add_message(message, level, source="MY_AI_UI")
        self.update_log_display(self.logging_manager.get_logs())

    def update_log_display(self, logs):
        self.log_display.setPlainText(logs)

    def format_conversation_history(self, messages):
        _messages = []
        for msg in messages:
            if not getattr(msg, "unspoken_message", False):
                _messages.append(msg)
        return _messages

    def get_reply(self, message, if_agent=False):
        self.log_message(f"Sending user message: {message}")
        _message = MessageContent(
            role="user",
            content=[ContentSegment(type="text", text=message)],
            timestamp=datetime.now()
        )
        
        message = format_messages([_message])
        self.update_chat([message])
        try:
            _response, _recent_conversations = asyncio.run(
                self.my_ai_assistant.process_and_create_chat_generation(
                message=_message,
                is_tool_call_permitted=if_agent
            )
            )
            self.log_message(f"Got response: {_response.content}")
            # history = self.format_conversation_history(_recent_conversations)
            history = format_messages([_response])
            self.update_chat(history)
            return history
        except Exception as e:
            self.log_message(f"Error getting response: {e}", "ERROR")
            return self.chat_history

    def update_chat(self, history):
        self.chat_list.clear()
        for msg in history:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = " ".join([seg.text for seg in content if hasattr(seg, "text")])
            item = QListWidgetItem(f"{role.capitalize()}: {content}")
            self.chat_list.addItem(item)
        self.chat_list.scrollToBottom()
        self.chat_history = history

    def send_message(self):
        msg = self.msg_input.text().strip()
        if msg:
            history = self.get_reply(msg, if_agent=False)
            
            self.update_chat(history)
            self.msg_input.clear()

    def send_agent_message(self):
        msg = self.msg_input.text().strip()
        if msg:
            history = self.get_reply(msg, if_agent=True)
            
            self.update_chat(history)
            self.msg_input.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyAIInterface()
    window.show()
    sys.exit(app.exec())
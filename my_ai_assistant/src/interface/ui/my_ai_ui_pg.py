import pygame
import sys
import threading
from datetime import datetime

from src.utils.my_ai_utils import format_messages
from src.memory_processor.memory_processor import MemoryProcessor
from src.core.api_server.data_models import ContentSegment, MessageContent
from src.inference_engine.inference_processor import InferenceProcessor
from src.utils.log_manager import LoggingManager
from src.core.my_ai_assistant import MyAIAssistant

pygame.init()

# --- Layout constants ---
WIDTH, HEIGHT = 1400, 800
BG_COLOR = (245, 245, 245)
PANEL_BG = (230, 230, 230)
BORDER = (180, 180, 180)
TEXT_COLOR = (30, 30, 30)
FONT = pygame.font.SysFont("consolas", 20)
SMALL_FONT = pygame.font.SysFont("consolas", 16)

MEDIA_W = 350
CHAT_W = 600
INFO_W = 350
PANEL_H = 600
FOOTER_H = 60
INPUT_H = 60
LOG_H = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MyAI Assistant (Pygame UI)")

def draw_panel(rect, label, content, font=FONT, lines=10):
    pygame.draw.rect(screen, PANEL_BG, rect)
    pygame.draw.rect(screen, BORDER, rect, 2)
    if label:
        label_surf = SMALL_FONT.render(label, True, TEXT_COLOR)
        screen.blit(label_surf, (rect.x + 8, rect.y + 4))
    # Draw content (multi-line)
    if content:
        y = rect.y + 28
        for i, line in enumerate(content.split("\n")[:lines]):
            line_surf = font.render(line, True, TEXT_COLOR)
            screen.blit(line_surf, (rect.x + 8, y))
            y += font.get_height() + 2

def draw_button(rect, text, active=True):
    color = (0, 180, 120) if active else (180, 180, 180)
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, BORDER, rect, 2, border_radius=8)
    txt = FONT.render(text, True, (255,255,255) if active else (80,80,80))
    txt_rect = txt.get_rect(center=rect.center)
    screen.blit(txt, txt_rect)

def draw_input(rect, text, active):
    pygame.draw.rect(screen, (255,255,255), rect)
    pygame.draw.rect(screen, (0, 120, 255) if active else BORDER, rect, 2)
    # Multi-line support
    lines = text.split("\n")
    for i, line in enumerate(lines):
        txt = FONT.render(line, True, TEXT_COLOR)
        screen.blit(txt, (rect.x+8, rect.y+8 + i*FONT.get_height()))

class MyAIPygameUI:
    def __init__(self):
        # Backend
        self.logging_manager = LoggingManager()
        self.memory_processor = MemoryProcessor()
        self.inference_processor = InferenceProcessor()
        self.my_ai_assistant = MyAIAssistant(
            memory_processor=self.memory_processor,
            inference_processor=self.inference_processor,
        )
        self.my_ai_assistant.speech_engine.tts_engine.init_screen()
        
        # self.visulizer_screen = self.my_ai_assistant.speech_engine.tts_engine.get_screen()
        self.video_content = self.my_ai_assistant.speech_engine.tts_engine.get_screen()
        
        self.enable_agent = False

        # UI state
        self.chat_history = []
        self.logs = "System Logs:\n[INFO] MyAIUI initialized successfully"
        # self.video_content = "Video feed placeholder"
        self.audio_content = "Audio visualizer placeholder"
        self.info_content = "Info panel placeholder"
        self.footer_content = "Footer info here"
        self.user_input = ""
        self.input_active = False
        self.running = True
        self.agent_btn_rect = pygame.Rect(20+MEDIA_W+20+CHAT_W-160, 20+PANEL_H+20+INPUT_H+20, 150, 40)
        self.input_rect = pygame.Rect(20+MEDIA_W+20, 20+PANEL_H+20, CHAT_W, INPUT_H)
        self.chat_rect = pygame.Rect(20+MEDIA_W+20, 20, CHAT_W, PANEL_H)
        self.log_rect = pygame.Rect(20+MEDIA_W+20+CHAT_W+20, 20+PANEL_H+20, INFO_W, LOG_H)
        self.scroll_offset = 0

        # Subscribe to logs
        self.logging_manager.subscribe(self.update_log_display)

    def log_message(self, message, level="INFO"):
        self.logging_manager.add_message(message, level, source="MY_AI_UI")
        self.logs = self.logging_manager.get_logs()

    def update_log_display(self, logs):
        self.logs = logs

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
        try:
            _response, _recent_conversations = self.my_ai_assistant.process_and_create_chat_generation(
                message=_message,
                is_tool_call_permitted=if_agent
            )
            self.log_message(f"Got response: {_response.content}")
            history = self.format_conversation_history(_recent_conversations)
            history = format_messages(history)
            self.chat_history = history
        except Exception as e:
            self.log_message(f"Error getting response: {e}", "ERROR")

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            screen.fill(BG_COLOR)
            # Layout
            media_rect = pygame.Rect(20, 20, MEDIA_W, PANEL_H)
            chat_rect = self.chat_rect
            info_rect = pygame.Rect(20+MEDIA_W+20+CHAT_W+20, 20, INFO_W, PANEL_H)
            input_rect = self.input_rect
            footer_rect = pygame.Rect(20+MEDIA_W+20, 20+PANEL_H+20+INPUT_H+10, CHAT_W, FOOTER_H)
            agent_btn_rect = self.agent_btn_rect
            log_rect = self.log_rect

            # Panels
            draw_panel(pygame.Rect(media_rect.x, media_rect.y, MEDIA_W, PANEL_H//2-10), "", self.video_content, lines=6)
            draw_panel(pygame.Rect(media_rect.x, media_rect.y+PANEL_H//2+10, MEDIA_W, PANEL_H//2-10), "", self.audio_content, lines=6)
            # Chatbot panel (scrollable)
            chat_lines = []
            for msg in self.chat_history[-15+self.scroll_offset:]:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    # If content is a list of ContentSegment
                    content = " ".join([seg.text for seg in content if hasattr(seg, "text")])
                chat_lines.append(f"{role.capitalize()}: {content}")
            draw_panel(chat_rect, "Chatbot", "\n".join(chat_lines), lines=15)
            # Info
            draw_panel(pygame.Rect(info_rect.x, info_rect.y, INFO_W, PANEL_H-LOG_H-10), "", self.info_content, lines=10)
            draw_panel(log_rect, "System Logs", self.logs, font=SMALL_FONT, lines=2)
            # Input
            draw_input(input_rect, self.user_input, self.input_active)
            # Footer
            draw_panel(footer_rect, "", self.footer_content, font=SMALL_FONT, lines=2)
            # Agent button
            draw_button(agent_btn_rect, "Ask MyAI Agent", active=True)

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if input_rect.collidepoint(event.pos):
                        self.input_active = True
                    else:
                        self.input_active = False
                    if agent_btn_rect.collidepoint(event.pos):
                        msg = self.user_input.strip()
                        if msg:
                            threading.Thread(target=self.get_reply, args=(msg, True)).start()
                            self.user_input = ""
                elif event.type == pygame.KEYDOWN and self.input_active:
                    if event.key == pygame.K_RETURN and (pygame.key.get_mods() & pygame.KMOD_SHIFT == 0):
                        msg = self.user_input.strip()
                        if msg:
                            threading.Thread(target=self.get_reply, args=(msg, False)).start()
                            self.user_input = ""
                    elif event.key == pygame.K_BACKSPACE:
                        self.user_input = self.user_input[:-1]
                    elif event.key == pygame.K_TAB:
                        self.user_input += "    "
                    elif event.key == pygame.K_UP:
                        self.scroll_offset = max(self.scroll_offset - 1, -len(self.chat_history)+15)
                    elif event.key == pygame.K_DOWN:
                        self.scroll_offset = min(self.scroll_offset + 1, 0)
                    else:
                        if len(self.user_input) < 300:
                            if event.unicode:
                                self.user_input += event.unicode

            pygame.display.flip()
            clock.tick(30)

if __name__ == "__main__":
    ui = MyAIPygameUI()
    ui.run()
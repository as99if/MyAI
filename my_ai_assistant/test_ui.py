# Main application entry point
from src.ui.my_ai_ui import MyAIInterface
    

if __name__ == "__main__":
    chat_interface = MyAIInterface()
    chat_interface.interface.launch(share=False, pwa=True)
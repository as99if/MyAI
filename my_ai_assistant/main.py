

# import json
import logging

# import multiprocessing
from src.interface.ui.my_ai_ui_qt import MyAIInterface
# from src.interface.ui.my_ai_ui_pg import MyAIPygameUI
# from src.interface.ui.my_ai_ui import MyAIInterface
    
# from src.core.my_ai import MyAI

logger = logging.getLogger(__name__)

# if __name__ == "__main__":
#     # if ui mode:
#     app = MyAIUI()
#     app.app_ui.launch(debug=True)
#     my_ai = MyAI()
#     computer = multiprocessing.Process(target=my_ai.__run__, name="computer")
#     computer.start()
#     computer.join()

import sys
# Main application entry point
from PyQt6.QtWidgets import QApplication
if __name__ == "__main__":
    # pygame UI
    # ui = MyAIPygameUI()
    # ui.run()
    
    # gradio
    # chat_interface = MyAIInterface()
    # chat_interface.interface.launch(share=False, pwa=True, inbrowser=True)
    
    # pyqt
    app = QApplication(sys.argv)
    window = MyAIInterface()
    window.show()
    sys.exit(app.exec())
# ui for logs and extended resultes and rendering of the agent's tasks

import gradio as gr
from src.utils.log_manager import LoggingManager

def create_log_viewer():
    # Initialize the singleton LoggingManager
    log_manager = LoggingManager()
    
    # Create state to store current logs
    state = gr.State("")
    
    def update_logs(current_logs):
        return current_logs
    
    # Subscribe to log updates
    def on_log_update(logs):
        if demo.is_running:
            state.value = logs

    # Subscribe to log updates
    log_manager.subscribe(on_log_update)

    with gr.Blocks(theme=gr.themes.Soft(
        primary_hue="gray",
        secondary_hue="gray",
        neutral_hue="gray",
        background_fill_primary="black",
    )) as demo:
        gr.Markdown("## System Logs")
        
        # Log display area with custom styling
        logs_display = gr.TextArea(
            value="",
            label="Logs",
            interactive=False,
            autoscroll=True,
            lines=20,
            elem_classes=["logs-display"],
        )

        # Add custom CSS
        gr.HTML("""
            <style>
                .logs-display textarea {
                    background-color: black !important;
                    color: #00ff00 !important;
                    font-family: 'Courier New', monospace !important;
                    font-size: 14px !important;
                }
            </style>
        """)

        # Update logs every second
        demo.load(update_logs, inputs=[state], outputs=[logs_display], every=1)

    return demo

if __name__ == "__main__":
    # do this when request from my_ai comes with show logs, or anything
    demo = create_log_viewer()
    demo.launch()


## TEST
"""_summary_
from utils.log_manager import LoggingManager
from ui.log_viewer import create_log_viewer
import time
import threading

def generate_test_logs():
    log_manager = LoggingManager()
    count = 0
    while True:
        log_manager.add_message(f"Test log message {count}", level="INFO")
        count += 1
        time.sleep(2)

if __name__ == "__main__":
    # Start log generation in a separate thread
    log_thread = threading.Thread(target=generate_test_logs, daemon=True)
    log_thread.start()
    
    # Launch the Gradio interface
    demo = create_log_viewer()
    demo.launch()
"""
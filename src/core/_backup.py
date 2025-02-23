import gradio as gr
import asyncio
from datetime import datetime
from typing import Tuple

class MyAIUI:
    def __init__(self, conversation_history_engine, inference_processor, voice_reply_enabled=False):
        self.conversation_history_engine = conversation_history_engine
        self.inference_processor = inference_processor
        self.voice_reply_enabled = voice_reply_enabled
        self.gui_interface = self.create_gradio_ui()

    def create_gradio_ui(self) -> gr.Blocks:
        """
        Creates and configures the Gradio web interface.

        Returns:
            gr.Blocks: Configured Gradio interface with chat and control components
        """
        def on_submit(message: str, history: list) -> Tuple[str, list]:
            # Simulate async behavior in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Prepare user message
            user_message = [{
                "role": "user", 
                "content": message, 
                "type": "user_message", 
                "timestamp": datetime.now().isoformat()
            }]
            
            # Get conversation history
            recent_conversation = []
            if self.conversation_history_engine:
                try:
                    recent_conversation = loop.run_until_complete(
                        self.conversation_history_engine.get_recent_conversation()
                    )
                    recent_conversation = recent_conversation + user_message
                    self.conversation_history_engine.add_conversation(user_message)
                except Exception as e:
                    print(f"Error getting recent conversation: {e}")
                    raise e
            else:
                recent_conversation = user_message
            
            # Get AI response
            response = loop.run_until_complete(
                self.inference_processor.create_chat_completion(recent_conversation)
            )
            loop.close()
            
            # Handle voice reply if enabled
            if self.voice_reply_enabled:
                spoken_reply, unspoken_reply = self.voice_reply(response)
                response_text = f"{spoken_reply}\n{unspoken_reply if unspoken_reply else ''}"
            else:
                response_text = response
                
            # Update conversation history
            history.append((message, response_text))
            return "", history

        def on_voice_toggle(value: bool):
            self.voice_reply_enabled = value
            return f"Voice reply {'enabled' if value else 'disabled'}"

        with gr.Blocks(title="MyAI Assistant") as interface:
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=600)
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False
                        )
                        submit = gr.Button("Send")
                
                with gr.Column(scale=1):
                    voice_toggle = gr.Checkbox(
                        label="Enable Voice Reply",
                        value=self.voice_reply_enabled
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        value="Voice reply disabled",
                        interactive=False
                    )
                    if True:  # if gui enabled
                        spectrogram = gr.Plot(label="Audio Spectrogram")

            msg.submit(
                on_submit,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            submit.click(
                on_submit,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            voice_toggle.change(
                on_voice_toggle,
                inputs=[voice_toggle],
                outputs=[status_text]
            )

        return interface
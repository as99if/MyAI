import asyncio
from datetime import datetime
import gradio as gr

class MyAIUI:
    def __init__(self, conversation_history_engine, inference_processor):
        self.conversation_history_engine = conversation_history_engine
        self.inference_processor = inference_processor
        self.stop_event = asyncio.Event()

    async def generate_response_async(self, prompt):
        print("----- generating async response -----")
        
        self.stop_event.clear()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Prepare user message
            user_message = [{
                "role": "user", 
                "content": prompt, 
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
                loop.close()
            else:
                recent_conversation = user_message
            
            
            response = await self.inference_processor.generate_chat_completion(recent_conversation)
            if self.stop_event.is_set():
                return "Generation stopped."
            return response
        except asyncio.CancelledError:
            return "Generation stopped."

    def generate_response(self, prompt):
        print("----- generating response -----")
        
        return asyncio.run(self.generate_response_async(prompt))

    def stop_generation(self):
        self.stop_event.set()

    def launch_ui(self):
        print("----- launching my ai ui -----")
        with gr.Blocks() as demo:
            gr.Markdown("# My AI Chatbot")
            chatbot = gr.Chatbot(height=600, type='messages')
            
            with gr.Row():
                message = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False
                )
                submit_button = gr.Button("Send")
                stop_button = gr.Button("Stop")

            submit_button.click(
                self.generate_response,
                inputs=message,
                outputs=chatbot
            )
            stop_button.click(self.stop_generation)

        demo.launch()

# Example usage:
# from inference_processor import InferenceProcessor
# inference_processor = InferenceProcessor()
# my_ai_ui = MyAIUI(conversation_history_engine, inference_processor)
# my_ai_ui.launch_ui()
"remove this"

import asyncio
import base64
from datetime import datetime
import multiprocessing

import os

import time
from typing import Any
from huggingface_hub import snapshot_download

from src.memory_processor.vision_history_engine import VisionHistoryEngine
from src.utils.download_from_hugginface import ModelDownloader
from src.vision_processor.object_detection_engine import ObjectDetectionEngine
from src.vision_processor.screenshot_service import take_screenshot
from src.utils.utils import load_config
from llama_cpp import Llama

class VisionInferenceEngine:
    def __init__(self):
        self.config = load_config()
        self.computer_vision = None
        self.vision_memory_engine = None
        self.object_detection_engine = None
        self.if_in_camera_loop = False
        # self.config = config
        self.model = None
        self.processor = None
        self.model_config = None
        self.if_model_loaded = False
        self.model_path = self.config.get("vlm_base_path")
        self.model_name = self.config.get('vlm') # paligemma2-3b-ft-docci-448-bf16, SmolVLM-500M-Instruct-bf16, mlx-community/SmolVLM-Instruct-bf16
        

    async def _initialize(self, vision_memory_engine: VisionHistoryEngine, object_detection_engine: ObjectDetectionEngine):
        self.if_model_loaded = False
        self.vision_memory_engine = vision_memory_engine
        self.object_detection_engine = object_detection_engine
        
        try:

            self.model = Llama(
                    model_path=f"{self.model_path}/{self.model_name}",
                    n_ctx=2048,
                    n_gpu_layers=-1,
                    chat_handler=True
                )

            
            # print(self.model)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        # self.model.get()
        
        # self.model_config = load_model_config(model_path)
        self.if_model_loaded = True
        return
        

    def generate_response(self, images: list[Any] | Any = None, prompt: str = "") -> Any:
        try:
            # Prepare input
            recent_messages = "" # self.conversation_history_engine.get_recent_messages()
            # TODO: split and process_recent_messages

            # images - image chunks of a video, from vision memory
            images = ["path/to/image1.jpg", "path/to/image2.jpg"]
            # prompt
            context = f"Here is some context of the user and the computer's conversation - \n{recent_messages}\n\nDescribe the images or chunk of a video. Do not say more than 100 words. Keep it concise."
            # Apply chat template
            
            formatted_prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                        {"type": "text", "text": context}
                    ]
                }
            ]

            self._add_to_memory(formatted_prompt)
            self.vision_memory.add_conversation([
                {
                    "role": "user", "content": formatted_prompt, type: "computer_vision_prompt", "timestamp": datetime.now().isoformat()
                }
            ])

            # Generate output
            
            response = self.model.create_chat_completion(
                messages=formatted_prompt,
                max_tokens=512,
                temperature=0.7
            )
            print(response)
            
            # Add response to memory
            self._add_to_memory({
                    "role": "computer_vision", "content": response[".."], type: "computer_vision_message", "timestamp": datetime.now().isoformat()
                })
            self.vision_memory.add_conversation([
                {
                    "role": "computer_vision", "content": response[".."], type: "computer_vision_message", "timestamp": datetime.now().isoformat()
                }
            ])
            

            # Add response to memory
            # self._add_to_memory(response.[".."][".."])
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            raise e

    def handle_video_prompt(self, video_chunk: list[Any] | Any, prompt: str = "") -> Any:
        # vidio chunk will is processed in vidio_processor
        try:
            response = self.generate_response(
                images=video_chunk, 
                prompt=prompt)
            # TODO: test
            print(response)

        except Exception as e:
            print(f"Error generating response: {e}")

    def handle_image_prompt(self, images: list = [], prompt: str = "") -> Any:
        try:
            response = self.generate_response(
                images=images,
                prompt=prompt
            )
            print(response)
        except Exception as e:
            print(f"Error generating response: {e}")

    def take_screenshot_and_generate(self, prompt: str = "") -> Any:
        images = []
        screenshot = take_screenshot()
        images.append(screenshot)

        try:
            response = self.generate_response(
                images=images,
                prompt=prompt
            )
            print(response)
            

        except Exception as e:
            print(f"Error generating response: {e}")

    

    def clear(self):
        self.model.clear()
        self.model_config.clear()
        self.processor.clear()
        self.computer_vision.terminate()
        self.computer_vision = None
        self.if_model_loaded = False
        self.if_in_camera_loop = False
        self.vision_memory_engine.cleanup()  # Clear vision memory

        
    def load(self):
        self.clear()
        self.computer_vision = multiprocessing.Process(
            target=self._initialize, name="computer-vision")
        self.computer_vision.start()
        self.computer_vision.join()
        # p.is_alive()


# m = MLXVisionEngine()
# asyncio.run(m._initialize())
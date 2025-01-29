"""
 MLXVisionEngine: A comprehensive vision processing engine using MLX and YOLO.

This engine provides capabilities for:
- Video/image processing and analysis
- Real-time camera feed processing
- Object detection with distance estimation
- Vision memory management
- Multi-modal (audio-visual) processing

Key Components:
-------------
- Vision Model: MLX-based vision model for image understanding
- Object Detection: YOLOv9 for real-time object detection (in models directory)
- Memory System: Vision history tracking and management
- Camera Interface: Real-time video capture and processing

Usage Examples:
-------------
1. Image Analysis:
    ```python
    engine = MLXVisionEngine(config)
    await engine._initialize()
    response = engine.handle_image_prompt(images=["image.jpg"], prompt="Describe this image")
    ```

2. Camera Stream:
    ```python
    engine = MLXVisionEngine(config)
    await engine._initialize()
    engine.start_camera_and_generate_loop(prompt="Monitor this scene", fps=4)
    ```

Methods:
-------
- _initialize(): Initialize vision model and memory systems
- handle_video_prompt(): Process video content with custom prompts
- handle_image_prompt(): Analyze static images
- take_screenshot_and_generate(): Capture and analyze screen content
- start_camera_and_generate_loop(): Begin real-time camera processing
- stop_camera_and_generate_loop(): End camera processing
- generate_response(): Core method for generating vision-based responses

Error Handling:
-------------
- Raises ModelLoadError: When vision model fails to load
- Raises CameraError: When camera access fails
- Raises MemoryError: When vision memory operations fail
- Raises RuntimeError: For general processing failures

Returns:
-------
Various methods return different types based on their function:
- Image analysis: Dict containing analysis results
- Camera stream: Generator yielding real-time analysis
- Memory operations: List of stored vision contexts

@author - Asif Ahmed - asif.shuvo2199@outlook.com
"""

import asyncio
from datetime import datetime
import multiprocessing
from aiomultiprocess import Pool
import os
import cv2
import time
from typing import Any
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config as load_model_config
from huggingface_hub import snapshot_download

from src.memory_processor.vision_history_engine import VisionHistoryEngine
from src.utils.download_from_hugginface import ModelDownloader
from src.vision_processor.object_detection_engine import ObjectDetectionEngine
from src.vision_processor.screenshot_service import take_screenshot
from src.utils.utils import load_config

class MLXVisionEngine:
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
        self.model_name = self.config.get('vlm')
        self.loop_vision_memory = [],
        self.memory_limit = 500

    async def _initialize(self):
        self.if_model_loaded = False
        self.vision_memory_engine = VisionHistoryEngine(self.config)
        # await self.vision_memory_engine.connect()

        model_path = f"src/vision_processor/models/{self.model_name}"

        try:
            try:
                self.model, self.processor = load(model_path)
            except Exception as e:
                _model_name = f"mlx-community/{self.model_name}"
                md = ModelDownloader(cache_dir="./models")
                md.download_model(_model_name, model_path)
                self.model, self.processor = load(model_path)
            
            # print(self.model)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        # self.model.get()
        
        self.model_config = load_model_config(model_path)
        self.if_model_loaded = True

    

    def _add_to_memory(self, response):
        """Add response to vision memory with size limit"""
        if len(self.vision_memory) > self.memory_limit - 2:
            self.loop_vision_memory.pop(0)  # Remove oldest entry
        self.loop_vision_memory.append(response)
        

    def generate_response(self, images: list[Any] | Any = None, prompt: str = "") -> Any:
        try:
            # Prepare input
            recent_messages = "" # self.conversation_history_engine.get_recent_messages()
            # TODO: split and process_recent_messages

            # images - image chunks of a video
            images = ["path/to/image1.jpg", "path/to/image2.jpg"]
            # prompt
            context = f"Here is some context of the user and the computer's conversation - \n{recent_messages}\n\nDescribe the images or chunk of a video. Do not say more than 100 words. Keep it concise."
            # Apply chat template
            formatted_prompt = apply_chat_template(
                self.processor, self.model_config, context, num_images=len(
                    images)
            )

            self._add_to_memory(formatted_prompt)
            self.vision_memory.add_conversation([
                {
                    "role": "user", "content": formatted_prompt, type: "computer_vision_prompt", "timestamp": datetime.now().isoformat()
                }
            ])

            # Generate output
            response = generate(self.model, self.processor,
                                formatted_prompt, images, verbose=True)
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

    def start_camera_and_generate_loop(self, prompt: str = "", range: int = 10, waiting_time: int = 20, fps: int = 4) -> Any:
        """
        Start camera and generate response in a loop
        :param prompt: prompt to generate response 
        :param range: number of frames to capture per iteration
        :param waiting_time: waiting time between captures in seconds
        :return: generator yielding responses
        """
        try:
            self.object_detection_engine = ObjectDetectionEngine()

            if fps <= 0:
                raise ValueError("frames_per_second must be positive")
            # Initialize camera
            self.video_capture_client = cv2.VideoCapture(0)
            if not self.video_capture_client.isOpened():
                raise RuntimeError("Failed to open camera")

            # Set camera properties
            self.video_capture_client.set(cv2.CAP_PROP_FPS, fps)
            self.video_capture_client.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture_client.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.if_in_camera_loop = True
            self.loop_vision_memory = []

            while self.if_model_loaded and self.if_in_camera_loop:
                images = []
                # start audio recording
                # Capture frames
                for _ in range(range):
                    ret, frame = self.video_capture_client.read()
                    if not ret:
                        continue

                    # Process frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame = cv2.resize(frame, (224, 224))  # Standard ML input size
                    _frame, detection_results = self.object_detection_engine.detect_objects_and_measure_distance(frame)
                    # visualise _frame in iframe if needed
                    
                    images.append(frame)
                    
                    if detection_results:
                        # Add detection results to vision memory
                        prompt = f"{prompt} \n\nObject Detection: \n {detection_results}"

                # finish audio recording, transcribe and add to prompt, add to prompt
                if images:
                    # Generate response using MLX model
                    try:

                        response = self.generate_response(
                            images=images,
                            prompt=prompt
                        )
                        # TODO: test
                        print(response)
                        self.vision_memory.add_conversation(
                            messages=[
                                {
                                    "role": "computer-vision", "content": response[".."][".."], type: "computer_message", "timestamp": datetime.now().isoformat()
                                }
                            ])

                    except Exception as e:
                        print(f"Error generating response: {e}")

        except Exception as e:
            print(f"Camera loop error: {e}")
            raise e
        finally:
            time.sleep(waiting_time)

    def stop_camera_and_generate_loop(self):
        
        if 'cap' in locals():
            self.cap.release()
        _loop_vision_memory = self.vision_memory
        self.clear()
        
        return _loop_vision_memory

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
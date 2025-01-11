import json
from src.inference_engine._inference_engine import InferenceEngine
from screenshot_service import take_screenshot
from siri_service import execute_siri_command
import speech_recognition as sr
from gtts import gTTS
import os
import playsound
import threading
from queue import Queue
from memory_service import MemoryService
from langchain.embeddings import OpenAIEmbeddings
from object_detection_engine import ObjectDetectionEngine
from datetime import datetime
from src.conversation_history_service import ConversationHistoryService
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from src.conversation_summarizer import ConversationSummarizer

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)
# Global flag to manage interruptions
user_interrupting = False


# Function to listen for user input
def listen(queue):
    global user_interrupting
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            user_interrupting = True  # User started speaking
            queue.put(text.lower())
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            queue.put(None)
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            queue.put(None)

# Function to convert text to speech and play it


def speak(text):
    # saves audio file and plays it
    # TODO: change it... make it parralel process to text generation, line by line... upon interruption, stop generation
    global user_interrupting
    try:
        tts = gTTS(text=text, lang='en')
        filename = "response.mp3"
        tts.save(filename)

        # Play audio in a separate thread to allow interruption detection
        def play_audio():
            global user_interrupting
            playsound.playsound(filename, block=True)
            if user_interrupting:  # Stop playing if interrupted
                os.remove(filename)

        audio_thread = threading.Thread(target=play_audio)
        audio_thread.start()
        audio_thread.join()  # Wait for the audio thread to complete or interrupt

        os.remove(filename)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")


async def run():
    # Initialize services
    vector_memory_service = MemoryService(
        embedding_model=OpenAIEmbeddings()
    )
    conversation_store_service = ConversationHistoryService()

    # Initialize inference engine with both memory services
    engine = InferenceEngine(
        config=config,
        vector_memory_service=vector_memory_service,
        conversation_store_service=conversation_store_service
    )

    # Initialize and start the summarizer in the background
    summarizer = ConversationSummarizer(engine, conversation_store_service)
    summarizer.start_summarization()
    print("Conversation summarization started in background...")

    # Initialize Object Detection Service
    object_detection_engine = ObjectDetectionEngine()

    try:
        global user_interrupting
        print("Voice Assistant is running...")

        while True:
            user_interrupting = False
            queue = Queue()

            listen_thread = threading.Thread(target=listen, args=(queue,))
            listen_thread.start()

            command = queue.get()

            if command:
                if "exit" in command:
                    print("Goodbye!")
                    break
                # all the elifs
                else:
                    # Get response from the selected API (with DSPy optimization)
                    # result = engine.infer(command)
                    result = await engine.advanced_async_inference(command)

                    # Respond to the user with AI's response
                    print(f"Assistant: {result['response']}")
                    speak(result["response"])

                """elif "take a screenshot" in command or "screenshot" in command:
                    print("Taking a screenshot...")
                    screenshot = take_screenshot()
                    if screenshot:
                        speak("Screenshot taken successfully.")
                        # Log this interaction into conversation history
                        # infer
                        #result = engine.infer(command, screenshot)
                        result = await engine.advanced_async_inference(command, screenshot)
                    else:
                        speak("Failed to take a screenshot.")
                
                elif "use siri" in command or "ask siri" in command:
                    siri_command = command.replace("use siri", "").replace("ask siri", "").strip()
                    if siri_command:
                        speak(f"Asking Siri: {siri_command}")
                        execute_siri_command(siri_command)
                        
                    else:
                        speak("Please specify what you want me to ask Siri.")
                
                # change implementation to have good ux
                elif "detect objects" in command or "what do you see" in command:
                    print("Detecting objects...")
                    speak("Looking for objects...")
                    
                    # Get detection results from the object detection engine
                    frame, detection_results = object_detection_engine.detect_objects_and_measure_distance()
                    
                    if detection_results:
                        # Format the detection results into a natural response
                        prompt = command + "\nThis is the result of the object detection from the camera: "
                        for i, (object_name, distance) in enumerate(detection_results):
                            if i > 0:
                                response += ", and " if i == len(detection_results) - 1 else ", "
                            response += f"a {object_name} about {distance:.1f} meters away"
                        prompt += "."
                        result = await engine.advanced_async_inference(prompt)
                        
                        
                        
                        # Speak the detection results
                        print(f"Assistant: {response}")
                        speak(response)

                    else:
                        response = "I don't see any objects I can recognize right now."
                        speak(response)"""

                

    finally:
        # Clean up resources
        summarizer.stop_summarization()

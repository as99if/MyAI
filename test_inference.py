import json
import pprint
from typing import List, Optional
import time
from src.core.clear_memory import clear_memory
from src.inference_engine.inference_engine import InferenceEngine
from src.memory.conversation_history_service import ConversationHistoryEngine

async def tests():
    # Test configuration
    config = {
        'api': 'llama_cpp_python',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_live_conversation_history_db': 4,
        'redis_password': None,
        "voice_reply_deactivated": True,
        "omni_mode": True,
        "groq_api_key": "your_groq_api_key_here",
        "groq_model_name": "",
        "enable_object_detection": False,
        "max_context_messages": 5,
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_memory_db": 0,
        "redis_complete_backup_conversation_history_db": 1,
        "conversation_backup_path": "",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "vector_dim": 1536
    }

    conversation_history_engine = ConversationHistoryEngine(config)
    await conversation_history_engine.connect()
    await conversation_history_engine.delete_all()
    sample_conversation_history_path = "/Users/asifahmed/Development/ProjectKITT/src/memory/test_conversation_history.json"
    try:
        with open(sample_conversation_history_path, "r") as f:
            sample_conversation_history = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError
    except json.JSONDecodeError:
        raise json.JSONDecodeError

    await conversation_history_engine.ingest_conversation_heistory(sample_conversation_history)

    inference_engine = InferenceEngine(
        config=config, conversation_history_service=conversation_history_engine)
    print("Testing chat completion...")
    
    
    start_time = time.time()
    response = await inference_engine.chat_completion("Hello, how are you today? Summarise my last five conversations in five bullet point sentences")
    print(response)
    reply = response['choices'][0]['message']['content']
    print(reply)
    end_time = time.time()
    print(f"took {end_time - start_time:.4f} seconds")
    
    clear_memory()

if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(tests())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")

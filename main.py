import json

from src.core.my_ai import MyAIAssistant
from src.inference_engine.inference_engine import InferenceEngine
from src.memory.conversation_history_service import ConversationHistoryEngine

with open("src/config.json", "r") as f:
    config = json.load(f)

if __name__ == "__main__":
    # check for conversation backup and summarize if needed in a different thread
    conveversation_history_service = ConversationHistoryEngine()
    innference_engine = InferenceEngine(config, conveversation_history_service)
    assistant = MyAIAssistant(config, innference_engine, conveversation_history_service) 
    
from langchain_ollama.chat_models import ChatOllama
import json
import pprint
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Optional
from src.memory.conversation_history_service import ConversationHistoryEngine

class InferenceEngine:
    
    def __init__(self, config, conversation_history_service):
        self.config = config
        self.conversation_history_service = conversation_history_service
        
        with open("src/prompts/system_prompts.json", "r") as f:
            self.system_prompt = json.load(f)
        
        # Maximum number of recent messages to include in context
        self.max_context_messages = config.get('max_context_messages', 5)
        
        self._initialize_llm_client()
        
    def _initialize_llm_client(self):
        """Initialize the appropriate LLM based on configuration"""
        if self.config.get("api") == "ollama":
            self.client = ChatOllama(
                model=self.config.get("ollama_model_name"),
                base_url=self.config.get("ollama_host", "http://localhost:11434"),
                timeout=45
            )
        else:
            raise ValueError("Invalid API specified in config.")
    
    def _prepare_context_messages(self, message: str) -> List:
        """
        Prepare context messages including system prompt and recent conversation history
        
        Args:
            message: Current user message
        
        Returns:
            List of messages to be used as context
        """
        # Start with system message
        messages = [SystemMessage(content=self.system_prompt["chatbot_system_prompt"])]
        print("system message")
        pprint.pprint(messages)
        # If conversation history service is available, retrieve recent messages
        if self.conversation_history_service:
            recent_conversations = self.conversation_history_service.get_recent_conversations(limit=self.max_context_messages)

            # json_recent_conversations = json.dumps(recent_conversations, sort_keys=True, indent=2)
            # print(json_recent_conversations)
            if recent_conversations:
                for msg in recent_conversations:
                    if msg['role'] == 'human':
                        messages.append(HumanMessage(content=msg['content']))
                    elif msg['role'] == 'assistant' or msg['role'] == 'ai':
                        messages.append(AIMessage(content=msg['content']))
        
        # Add current user message
        messages.append(HumanMessage(content=message))
        
        return messages
    
    def chat_completion(self, message: str) -> str:
        """
        Generate chat completion with context management
        
        Args:
            message: User's input message
        
        Returns:
            AI's response
        """
        # Prepare context messages
        messages = self._prepare_context_messages(message)
        
        try:
            # Generate response
            print("\n--------------------------------\n")
            pprint.pprint(messages, indent=4)
            response = self.client.invoke(messages)
            response_text = response.content
            print('resposnesss')
            pprint.pprint(response, indent=4)
            
            
            return response_text
        
        except Exception as e:
            raise Exception(f"Error during chat completion: {e}")
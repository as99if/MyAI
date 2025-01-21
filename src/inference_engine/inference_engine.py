from datetime import datetime
import json
import pprint
from typing import List, Optional

from llama_cpp import Llama as LlamaCPP
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
import requests


class InferenceEngine:

    def __init__(self, config, conversation_history_service):
        self.config = config
        self.conversation_history_service = conversation_history_service

        with open("src/prompts/system_prompts.json", "r") as f:
            self.system_prompt = json.load(f)

        # Maximum number of recent messages to include in context
        self.max_context_messages = config.get('max_context_messages', 10)
        self.max_context_messages_day_limit = config.get('max_context_messages_day_limit', 5)
        self.base_url = None

        self._initialize_llm_client()

    def _initialize_llm_client(self):
        """Initialize the appropriate LLM based on configuration"""
        if self.config.get("api") == "llama_cpp_python":
            # omni model and ollama implementation not optimal yet, for it
            # probably will remove ollama and langchain from here
            self.client = LlamaCPP(
                model_path="/Users/asifahmed/Development/ProjectKITT/llm/base_model/Llama-3.2-3B-Instruct-Q8_0.gguf",
                chat_format="chatml-function-calling",
                # chat_handler=chat_handler, # if mllm
                n_gpu_layers=-1,  # Offload all layers to GPU
                n_ctx=2048,       # Reasonable context length for 16GB RAM
                n_batch=256,      # Batch size for prompt processing
                n_threads=4,      # Utilize M1 cores effectively
                n_threads_batch=4,# Match with n_threads for batch processing
                f16_kv=True,      # Use half precision for key/value cache
                use_mmap=True,    # Enable memory mapping
                use_mlock=True,   # Keep model in RAM
                flash_attn=True,  # Enable flash attention for faster processing
                offload_kqv=True, # Offload K,Q,V matrices to GPU
                verbose=False,
                # draft_model=LlamaPromptLookupDecoding(num_pred_tokens=5)
            )

        else:
            raise ValueError("Invalid API specified in config.")

    async def _prepare_context_messages(self, message: str) -> List:
        """
        Prepare context messages including system prompt and recent conversation history

        Args:
            message: Current user message

        Returns:
            List of messages to be used as context
        """
        # Start with system message
        messages = []

        # If conversation history service is available, retrieve recent messages
        if self.conversation_history_service:
            recent_conversations = await self.conversation_history_service.get_recent_conversations(
                days=5, limit=self.max_context_messages
            )
            return recent_conversations

    async def chat_completion(self, message: str) -> str:
        """
        Generate chat completion with context management

        Args:
            message: User's input message

        Returns:
            AI's response
        """
        # Prepare context messages
        messages = [{
            "role": "system",
            "content": f"{self.system_prompt['chatbot_system_prompt']} {self.system_prompt['chatbot_guidelines']}",
        }]
        context_recent_onversations = await self._prepare_context_messages(message)
        # print("context messages:\n", json.dumps(messages, indent=2))
        messages.extend(context_recent_onversations)
        messages.append({
            "role": "user",
            "content": message,
        })
        print("processed prompt:\n", json.dumps(messages, indent=2))

        try:
            if self.config.get("api") == "llama_cpp_python":
                response = self.client.create_chat_completion(
                    messages=messages,
                    max_tokens=2048
                    # temperature=.9,
                    # response_format={
                    #     "type": "json_object",
                    #     "schema": {
                    #         "type": "object",
                    #         "properties": {"team_name": {"type": "string"}},
                    #         "required": ["team_name"],
                    #     },
                    # },
                    # tools=[{
                    #     "type": "function",
                    #     "function": {
                    #         "name": "UserDetail",
                    #         "parameters": {
                    #             "type": "object",
                    #             "title": "UserDetail",
                    #             "properties": {
                    #                 "name": {
                    #                     "title": "Name",
                    #                     "type": "string"
                    #                 },
                    #                 "age": {
                    #                     "title": "Age",
                    #                     "type": "integer"
                    #                 }
                    #             },
                    #             "required": ["name", "age"]
                    #         }
                    #     }
                    # }],
                    # tool_choice={
                    #     "type": "function",
                    #     "function": {
                    #         "name": "UserDetail"
                    #     }
                    # }

                )

                return response



        except Exception as e:
            raise Exception(f"Error during chat completion: {e}")







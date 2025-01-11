"""import os
import asyncio
import dspy
from langchain_groq.chat_models import ChatGroq
from langchain_ollama.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from typing import Optional, Union, Dict, Any
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


class _InferenceEngine:
    def __init__(self, config: Dict[str, Any], vector_memory_service, conversation_store_service):
        
        self.api = config["api"]
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.deepseek_api_key = os.getenv("DEEPSEEK")
        self.ollama_host = config.get("ollama_host", "http://localhost:11434")
        self.use_memory = config["use_memory"]
        self.optimize_prompt = config["optimize_prompt"]
        self.conversation_store_service = conversation_store_service

        # Memory services
        self.vector_memory = vector_memory_service

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Initialize LLM
        self._initialize_llm()

        # Initialize memory and conversation components
        self._initialize_memory_components()

        # Initialize prompt optimizer if enabled
        if self.optimize_prompt:
            self._initialize_prompt_optimizer()

    async def infer_async(self, prompt: Union[str, dict], image: Optional[Image.Image] = None, encoded_image: str = None):
        """
        Asynchronous inference method
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.infer,
            prompt,
            image,
            encoded_image
        )

    async def _get_relevant_context_async(self, prompt: str) -> str:

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._get_relevant_context,
            prompt
        )

    async def _optimize_prompt_async(self, context_enhanced_prompt: str) -> str:

        if not self.optimize_prompt:
            return context_enhanced_prompt

        try:
            loop = asyncio.get_event_loop()
            optimized = await loop.run_in_executor(
                self.executor,
                self._optimize_prompt,
                context_enhanced_prompt
            )
            return optimized
        except Exception as e:
            print(f"Async prompt optimization failed: {e}")
            return context_enhanced_prompt

    def _optimize_prompt(self, prompt: str) -> str:

        try:
            optimized = self.optimizer(prompt)
            return optimized.output
        except Exception as e:
            print(f"Prompt optimization failed: {e}")
            return prompt

    async def _handle_image_inference_async(self, prompt_text: str, image: Optional[Image.Image], encoded_image: Optional[str]):

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._handle_image_inference,
            prompt_text,
            image,
            encoded_image
        )

    async def _handle_text_inference_async(self, prompt_text: str):

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._handle_text_inference,
            prompt_text
        )

    async def advanced_async_inference(self, prompt: Union[str, dict], image: Optional[Image.Image] = None, encoded_image: str = None):
        
        try:
            # Convert prompt to string if needed
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)

            # Get context before optimization for better relevance
            context_enhanced_prompt = await self._get_relevant_context_async(prompt_text)

            # Optimize prompt if enabled
            final_prompt = await self._optimize_prompt_async(context_enhanced_prompt)

            # Prepare metadata for tracking
            metadata = {
                "has_image": bool(image or encoded_image),
                "prompt_length": len(prompt_text),
                "context_length": len(context_enhanced_prompt),
                "timestamp": datetime.now().isoformat()
            }

            # Handle inference based on input type
            if image or encoded_image:
                response = await self._handle_image_inference_async(final_prompt, image, encoded_image)
            else:
                response = await self._handle_text_inference_async(final_prompt)

            # Save interaction with enhanced metadata
            self._save_interaction(prompt_text, response["response"], metadata)

            return {
                "user_prompt": prompt_text,
                "optimized_prompt": final_prompt if self.optimize_prompt else None,
                "response": response["response"],
                "chat_history": self.memory.chat_memory.messages,
                "metadata": metadata
            }

        except Exception as e:
            print(f"Async Inference failed: {e}")
            return {"error": str(e)}

    def _initialize_llm(self):
        # TODO: hyperparameter optimisations
        """Initialize the appropriate LLM based on configuration"""
        if self.api == "groq":
            self.llm = ChatGroq(model="llama-3.3-70b-specdec",
                                api_key=self.groq_api_key, cache=True)
        elif self.api == "ollama":
            self.llm = ChatOllama(base_url=self.ollama_host)
        else:
            raise ValueError("Invalid API specified in config.")

    def _initialize_memory_components(self):
        """Initialize conversation memory components"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )

    def _initialize_prompt_optimizer(self):
        """Initialize the DSPy prompt optimizer"""
        if self.api == "groq":
            dspy.settings.configure(lm=dspy.Groq(api_key=self.groq_api_key))
        elif self.api == "ollama":
            dspy.settings.configure(lm=dspy.Ollama(base_url=self.ollama_host))

        class PromptOptimizer(dspy.Signature):
            """Optimize the input prompt for better results."""
            input = dspy.InputField(desc="Original user prompt")
            output = dspy.OutputField(desc="Optimized prompt")

        self.optimizer = dspy.Predict(PromptOptimizer)

    def _get_relevant_context(self, prompt: str) -> str:
        """Get relevant context from both memory services"""
        if not self.use_memory:
            return prompt

        # Get semantic context from vector store with metadata filtering
        vector_context = self.vector_memory.retrieve_relevant_context(
            prompt,
            top_k=2  # Reduced from 3 for better focus
        )

        # Get recent conversation history from Redis
        recent_conversations = self.conversation_store_service.get_recent_conversations(
            limit=10)

        # Get knowledge base entries
        knowledge_entries = self.vector_memory.search_knowledge_base(
            query=prompt,
            limit=2
        )

        # Build context intelligently
        context_parts = []

        # Add recent conversations if relevant
        if recent_conversations:
            conv_context = "Recent conversation context:\n"
            for conv in recent_conversations:
                # Only last 2 messages per conversation
                for msg in conv.get('messages', [])[-2:]:
                    conv_context += f"{msg['role']}: {msg['content']}\n"
            context_parts.append(conv_context)

        # Add vector memory context if available
        if vector_context:
            memory_context = "Relevant past interactions:\n"
            memory_context += "\n".join(
                [mem.page_content for mem in vector_context])
            context_parts.append(memory_context)

        # Add knowledge base entries if available
        if knowledge_entries:
            knowledge_context = "Relevant knowledge:\n"
            for entry in knowledge_entries:
                if entry['metadata'].get('type') == 'image_data':
                    knowledge_context += f"From image {entry['metadata'].get('filename', 'unknown')}: "
                knowledge_context += f"{entry['content']}\n"
            context_parts.append(knowledge_context)

        # Combine contexts with clear separation
        if context_parts:
            full_context = "\n\n".join(context_parts)
            return f"{full_context}\n\nCurrent query: {prompt}"

        return prompt

    def _save_interaction(self, prompt: str, response: str, metadata: Dict = None):
        """Save interaction to both memory services with optimized metadata"""
        if not self.use_memory:
            return

        # Prepare enhanced metadata
        enhanced_metadata = {
            "timestamp": datetime.now().isoformat(),
            "optimized": self.optimize_prompt,
            "llm_provider": self.api,
            **(metadata or {})
        }

        # Save to vector store for semantic search
        self.vector_memory.save_interaction(prompt, response)

        # Save to Redis for conversation history
        chat_messages = [
            ("user", prompt),
            ("assistant", response)
        ]
        self.conversation_store_service.save_chat_messages(
            chat_messages, enhanced_metadata)

        # Update conversation buffer memory
        self.memory.save_context({"input": prompt}, {"output": response})

    def infer(self, prompt: Union[str, dict], image: Optional[Image.Image] = None, encoded_image: str = None):
        """
        Perform inference with optional image input and conversation memory.
        """
        try:
            # Convert prompt to string if needed
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)

            # Get context before optimization for better relevance
            context_enhanced_prompt = self._get_relevant_context(prompt_text)

            # Optimize prompt if enabled
            if self.optimize_prompt:
                try:
                    optimized = self.optimizer(context_enhanced_prompt)
                    final_prompt = optimized.output
                except Exception as e:
                    print(
                        f"Prompt optimization failed: {e}. Using original prompt.")
                    final_prompt = context_enhanced_prompt
            else:
                final_prompt = context_enhanced_prompt

            # Prepare metadata for tracking
            metadata = {
                "has_image": bool(image or encoded_image),
                "prompt_length": len(prompt_text),
                "context_length": len(context_enhanced_prompt),
                "timestamp": datetime.now().isoformat()
            }

            # Handle inference based on input type
            if image or encoded_image:
                response = self._handle_image_inference(
                    final_prompt, image, encoded_image)
            else:
                response = self._handle_text_inference(final_prompt)

            # Save interaction with enhanced metadata
            self._save_interaction(prompt_text, response["response"], metadata)

            return {
                "user_prompt": prompt_text,
                "optimized_prompt": final_prompt if self.optimize_prompt else None,
                "response": response["response"],
                "chat_history": self.memory.chat_memory.messages,
                "metadata": metadata
            }

        except Exception as e:
            print(f"Inference failed: {e}")
            return {"error": str(e)}

    def _handle_image_inference(self, prompt_text: str, image: Optional[Image.Image], encoded_image: Optional[str]):
        """Handle inference with image input"""
        image_content = {
            "type": "image_url",
            "image_url": image if image else {"url": f"data:image/jpeg;base64,{encoded_image}"}
        }

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                image_content
            ]
        )

        # TODO: add system instructions at the beginning ot the message
        response = self.llm.invoke([message])
        response_text = response.content if hasattr(
            response, 'content') else str(response)

        return {
            "user_prompt": prompt_text,
            "response": response_text,
            "chat_history": self.memory.chat_memory.messages
        }

    def _handle_text_inference(self, prompt_text: str):
        """Handle text-only inference"""
        try:
            # TODO: add system instructions at the beginning ot the message
            response = self.conversation_chain({"question": prompt_text})
            return {
                "user_prompt": prompt_text,
                "response": response['answer'],
                "chat_history": self.memory.chat_memory.messages
            }
        except Exception as e:
            print(
                f"Conversation chain failed: {e}. Falling back to direct LLM call.")
            response = self.llm.invoke([HumanMessage(content=prompt_text)])
            return {
                "user_prompt": prompt_text,
                "response": response.content if hasattr(response, 'content') else str(response),
                "chat_history": self.memory.chat_memory.messages
            }

"""

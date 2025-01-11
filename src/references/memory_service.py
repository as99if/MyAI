import numpy as np
import redis
from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Union, Dict
import base64
from PIL import Image
import io
import pytesseract
from datetime import datetime
import json
import os
import aioredis
import asyncio

class MemoryService:
    def __init__(self, embedding_model):
        """
        Initialize Redis with vector similarity search capability
        
        Args:
            embedding_model: Model to generate embeddings (e.g., OpenAIEmbeddings)
        """
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize Redis client with config
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0),
            password=config.get('redis_password', None),
            decode_responses=True
        )
        
        self.embedding_model = embedding_model
        self.vector_dim = 1536  # OpenAI embedding dimension
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('chunk_size', 1000),
            chunk_overlap=config.get('chunk_overlap', 200),
            length_function=len,
        )
        
        # Initialize Redis indexes
        self._create_indexes()

    def _create_indexes(self):
        """Create Redis indexes for vector search"""
        try:
            # Conversation memory index
            conversation_schema = (
                VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": self.vector_dim, "DISTANCE_METRIC": "COSINE"}),
                TextField("text"),
                TagField("type"),
                TextField("timestamp")
            )
            
            try:
                self.redis_client.ft("conversation_idx").dropindex()
            except:
                pass
                
            self.redis_client.ft("conversation_idx").create_index(
                fields=conversation_schema,
                definition=IndexDefinition(prefix=["conv:"], index_type=IndexType.HASH)
            )
            
            # Knowledge base index
            knowledge_schema = (
                VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": self.vector_dim, "DISTANCE_METRIC": "COSINE"}),
                TextField("text"),
                TagField("type"),
                TextField("metadata"),
                TextField("timestamp")
            )
            
            try:
                self.redis_client.ft("knowledge_idx").dropindex()
            except:
                pass
                
            self.redis_client.ft("knowledge_idx").create_index(
                fields=knowledge_schema,
                definition=IndexDefinition(prefix=["know:"], index_type=IndexType.HASH)
            )
            
        except Exception as e:
            print(f"Error creating Redis indexes: {e}")
            raise

    async def save_interaction_async(self, user_prompt: str, ai_response: str):
        """Asynchronous save of interaction"""
        async with aioredis.from_url(self.redis_url) as redis:
            timestamp = datetime.now().isoformat()
            
            # Async embedding generation
            user_embedding = await self._get_embedding_async(user_prompt)
            ai_embedding = await self._get_embedding_async(ai_response)
            
            # Async Redis operations
            await redis.hset(f"conv:user:{timestamp}", mapping={
                "text": user_prompt,
                "type": "user_message",
                "timestamp": timestamp,
                "embedding": user_embedding
            })
            
            await redis.hset(f"conv:ai:{timestamp}", mapping={
                "text": ai_response,
                "type": "ai_response",
                "timestamp": timestamp,
                "embedding": ai_embedding
            })

    async def _get_embedding_async(self, text: str) -> List[float]:
        """Async embedding generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.embedding_model.embed_query, 
            text
        )

    def save_interaction(self, user_prompt: str, ai_response: str):
        """Save conversation interactions"""
        timestamp = datetime.now().isoformat()
        
        # Save user prompt
        user_embedding = self._get_embedding(user_prompt)
        user_key = f"conv:user:{timestamp}"
        self.redis_client.hset(user_key, mapping={
            "text": user_prompt,
            "type": "user_message",
            "timestamp": timestamp,
            "embedding": np.array(user_embedding, dtype=np.float32).tobytes()
        })
        
        # Save AI response
        ai_embedding = self._get_embedding(ai_response)
        ai_key = f"conv:ai:{timestamp}"
        self.redis_client.hset(ai_key, mapping={
            "text": ai_response,
            "type": "ai_response",
            "timestamp": timestamp,
            "embedding": np.array(ai_embedding, dtype=np.float32).tobytes()
        })

    def upload_text_data(self, text: str, metadata: Optional[dict] = None) -> bool:
        """Upload and vectorize text data into knowledge base"""
        try:
            chunks = self.text_splitter.split_text(text)
            timestamp = datetime.now().isoformat()
            
            for i, chunk in enumerate(chunks):
                chunk_embedding = self._get_embedding(chunk)
                chunk_key = f"know:text:{timestamp}:{i}"
                
                chunk_metadata = {
                    "type": "text_data",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "upload_time": timestamp,
                    **(metadata or {})
                }
                
                self.redis_client.hset(chunk_key, mapping={
                    "text": chunk,
                    "type": "text_data",
                    "metadata": json.dumps(chunk_metadata),
                    "timestamp": timestamp,
                    "embedding": np.array(chunk_embedding, dtype=np.float32).tobytes()
                })
            
            return True
            
        except Exception as e:
            print(f"Error uploading text data: {e}")
            return False

    def upload_image_data(self, 
                         image: Union[Image.Image, str, bytes], 
                         metadata: Optional[dict] = None) -> bool:
        """Upload and vectorize image data, including extracted text"""
        try:
            # Convert input to PIL Image
            if isinstance(image, str):
                image_data = base64.b64decode(image)
                img = Image.open(io.BytesIO(image_data))
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image))
            elif isinstance(image, Image.Image):
                img = image
            else:
                raise ValueError("Unsupported image format")
            
            # Extract text from image
            extracted_text = pytesseract.image_to_string(img)
            timestamp = datetime.now().isoformat()
            
            if extracted_text.strip():
                chunks = self.text_splitter.split_text(extracted_text)
                for i, chunk in enumerate(chunks):
                    chunk_embedding = self._get_embedding(chunk)
                    chunk_key = f"know:image:{timestamp}:{i}"
                    
                    chunk_metadata = {
                        "type": "image_data",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "has_extracted_text": True,
                        "upload_time": timestamp,
                        **(metadata or {})
                    }
                    
                    self.redis_client.hset(chunk_key, mapping={
                        "text": chunk,
                        "type": "image_data",
                        "metadata": json.dumps(chunk_metadata),
                        "timestamp": timestamp,
                        "embedding": np.array(chunk_embedding, dtype=np.float32).tobytes()
                    })
            
            return True
            
        except Exception as e:
            print(f"Error uploading image data: {e}")
            return False

    def search_knowledge_base(self, query: str, limit: int = 5) -> List[dict]:
        """Search uploaded data in knowledge base"""
        try:
            query_embedding = self._get_embedding(query)
            vector_query = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Prepare Redis vector similarity query
            q = Query(f"*=>[KNN {limit} @embedding $vector AS score]")\
                .sort_by("score")\
                .return_fields("text", "metadata", "score")\
                .dialect(2)
            
            # Execute search
            results = self.redis_client.ft("knowledge_idx").search(
                q,
                query_params={"vector": vector_query}
            )
            
            return [
                {
                    "content": doc.text,
                    "metadata": json.loads(doc.metadata),
                    "score": doc.score
                }
                for doc in results.docs
            ]
            
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []

    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[dict]:
        """Retrieve most relevant past interactions"""
        try:
            query_embedding = self._get_embedding(query)
            vector_query = np.array(query_embedding, dtype=np.float32).tobytes()
            
            q = Query(f"*=>[KNN {top_k} @embedding $vector AS score]")\
                .sort_by("score")\
                .return_fields("text", "type", "timestamp", "score")\
                .dialect(2)
            
            results = self.redis_client.ft("conversation_idx").search(
                q,
                query_params={"vector": vector_query}
            )
            
            return [
                {
                    "page_content": doc.text,
                    "metadata": {
                        "type": doc.type,
                        "timestamp": doc.timestamp,
                        "score": doc.score
                    }
                }
                for doc in results.docs
            ]
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

    def augment_prompt_with_memory(self, original_prompt: str) -> str:
        """Enhance prompt with relevant historical context and knowledge"""
        # Get conversation context
        relevant_conversations = self.retrieve_relevant_context(original_prompt)
        conversation_context = "\n".join([
            mem["page_content"] for mem in relevant_conversations
        ])
        
        # Get relevant knowledge
        relevant_knowledge = self.search_knowledge_base(original_prompt, limit=2)
        knowledge_context = "\n".join([
            f"Related information: {item['content']}"
            for item in relevant_knowledge
        ])
        
        # Combine contexts
        augmented_prompt = f"""
        Context from past conversations:
        {conversation_context}
        
        Relevant knowledge:
        {knowledge_context}

        Current Query: {original_prompt}
        """
        return augmented_prompt
    
    
    
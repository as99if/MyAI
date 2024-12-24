import redis
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
from uuid import uuid4
import os

class ConversationHistoryService:
    def __init__(self, connection_params: Dict[str, Any]):
        """
        Initialize Redis connection
        
        Args:
            connection_params: Dictionary containing Redis connection parameters
                             (host, port, db, password, etc.)
        """
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Initialize Redis client with config
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_conversation_history_db', 0),
            password=config.get('redis_password', None),
            decode_responses=True
        )

    def save_conversation_entry(self, entry_type: str, text: str):
        """Save a single conversation entry"""
        entry = {
            'type': entry_type,
            'datetime': datetime.now().isoformat(),
            'text': text
        }
        
        # Store in a Redis list with key 'conversation_history'
        self.redis_client.lpush('conversation_history', json.dumps(entry))
        
        # Optional: Trim the list to keep only recent entries (e.g., last 1000)
        self.redis_client.ltrim('conversation_history', 0, 999)

    def save_chat_messages(self, messages: List[Tuple[str, str]], metadata: Dict = None):
        """
        Save a list of chat messages
        
        Args:
            messages: List of (role, content) tuples
            metadata: Optional dictionary of metadata for the conversation
        """
        conversation_id = str(uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create conversation data structure
        conversation = {
            'id': conversation_id,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'messages': [
                {
                    'role': role,
                    'content': content,
                    'timestamp': timestamp
                } for role, content in messages
            ]
        }
        
        # Store the conversation in Redis
        # Use a hash to store conversation details
        conversation_key = f'chat:{conversation_id}'
        self.redis_client.hset(
            conversation_key,
            mapping={
                'data': json.dumps(conversation),
                'timestamp': timestamp
            }
        )
        
        # Add to a sorted set for easy retrieval by timestamp
        self.redis_client.zadd(
            'conversations_by_time',
            {conversation_id: datetime.fromisoformat(timestamp).timestamp()}
        )
        
        # Optional: Set expiration for conversation data (e.g., 30 days)
        self.redis_client.expire(conversation_key, 60 * 60 * 24 * 30)

    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent conversations"""
        # Get recent conversation IDs from sorted set
        recent_ids = self.redis_client.zrevrange('conversations_by_time', 0, limit - 1)
        
        conversations = []
        for conv_id in recent_ids:
            conv_key = f'chat:{conv_id}'
            conv_data = self.redis_client.hget(conv_key, 'data')
            if conv_data:
                conversations.append(json.loads(conv_data))
        
        return conversations

    def get_conversation_history(self, limit: int = None) -> List[Dict]:
        """
        Retrieve conversation history entries
        
        Args:
            limit: Optional limit on number of entries to retrieve
        """
        try:
            if limit:
                entries = self.redis_client.lrange('conversation_history', 0, limit - 1)
            else:
                entries = self.redis_client.lrange('conversation_history', 0, -1)
            return [json.loads(entry) for entry in entries]
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []

    def clear_old_conversations(self, days_to_keep: int = 30):
        """Clear conversations older than specified days"""
        cutoff_timestamp = (
            datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        )
        
        # Get old conversation IDs
        old_ids = self.redis_client.zrangebyscore(
            'conversations_by_time',
            '-inf',
            cutoff_timestamp
        )
        
        if old_ids:
            # Remove from sorted set
            self.redis_client.zremrangebyscore(
                'conversations_by_time',
                '-inf',
                cutoff_timestamp
            )
            
            # Remove conversation data
            for conv_id in old_ids:
                self.redis_client.delete(f'chat:{conv_id}') 
                
    
    def replace_conversations(self, start_index: int, end_index: int, new_entry: Dict):
        """
        Replace a range of conversations with a new entry (e.g., a summary)
        
        Args:
            start_index: Starting index to replace
            end_index: Ending index to replace
            new_entry: New entry to insert (usually a summary)
        """
        try:
            # Get all current entries
            entries = self.get_conversation_history()
            
            # Create new list with summary replacing old entries
            updated_entries = (
                [new_entry] +  # Add summary at the beginning
                entries[end_index:]  # Keep all entries after the replaced ones
            )
            
            # Clear the existing list
            self.redis_client.delete('conversation_history')
            
            # Add updated entries back to Redis
            for entry in reversed(updated_entries):  # Reverse to maintain correct order
                self.redis_client.lpush('conversation_history', json.dumps(entry))
                
            return True
            
        except Exception as e:
            print(f"Error replacing conversations: {e}")
            return False
    
    
    
    
import os
import aioredis
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

class ConversationHistoryEngine:
    async def __init__(self, config):
        self.config = config
        # Initialize Redis client with config
        self.redis_client = await aioredis.create_redis_pool(
            (config.get('redis_host', 'localhost'),
             config.get('redis_port', 6379)),
            db=config.get('redis_live_conversation_history_db', 0),
            password=config.get('redis_password', None),
            encoding='utf-8'
        )
        
        # Configure persistence
        await self.redis_client.config_set('save', '900 1 300 10 60 10000')
        await self.redis_client.config_set('appendonly', 'yes')
        await self.redis_client.config_set('appendfsync', 'everysec')

    async def save_conversation_entry(self, entry_type: str, text: str):
        timestamp = datetime.now()
        date_key = timestamp.strftime('%Y-%m-%d')
        time_key = timestamp.strftime('%H:%M:%S')
        
        entry = {
            'time': time_key,
            'type': entry_type,
            'content': text,
            'timestamp': timestamp.isoformat()
        }
        
        daily_chat_key = f'chat:day:{date_key}'
        
        # Get existing entries for the day
        daily_entries = await self.redis_client.hget(daily_chat_key, 'data')
        if daily_entries:
            daily_entries = json.loads(daily_entries)
            daily_entries.append(entry)
        else:
            daily_entries = [entry]
        
        # Store updated daily entries
        await self.redis_client.hset(
            daily_chat_key,
            'data', json.dumps(daily_entries),
            'last_updated', timestamp.isoformat()
        )
        
        # Add to time-sorted index
        await self.redis_client.zadd(
            'chat_days',
            timestamp.timestamp(),
            date_key
        )
        
        # Set expiry for 30 days
        await self.redis_client.expire(daily_chat_key, 60 * 60 * 24 * 30)

    async def save_chat_segment(self, messages: List[Tuple[str, str]], metadata: Dict = None):
        timestamp = datetime.now()
        date_key = timestamp.strftime('%Y-%m-%d')
        time_key = timestamp.strftime('%H:%M:%S')
        
        conversation = [
                {
                    'role': role,
                    'content': content,
                    'timestamp': timestamp.isoformat()
                } for role, content in messages
            ]
        
        daily_chat_key = f'chat:day:{date_key}'
        
        # Get existing conversations for the day or create new list
        daily_chats = await self.redis_client.hget(daily_chat_key, 'data')
        if daily_chats:
            daily_chats = json.loads(daily_chats)
            daily_chats.append(conversation)
        else:
            daily_chats = [conversation]
        
        # Store updated daily conversations
        await self.redis_client.hset(
            daily_chat_key,
            'data', json.dumps(daily_chats),
            'last_updated', timestamp.isoformat()
        )
        
        # Add to time-sorted index
        await self.redis_client.zadd(
            'chat_days',
            timestamp.timestamp(),
            date_key
        )
        
        # Set expiry for 30 days
        await self.redis_client.expire(daily_chat_key, 60 * 60 * 24 * 30)
    
    async def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        recent_ids = await self.redis_client.zrevrange('conversations_by_time', 0, limit - 1)
        
        conversations = []
        for conv_id in recent_ids:
            conv_key = f'chat:{conv_id}'
            conv_data = await self.redis_client.hget(conv_key, 'data')
            if conv_data:
                conversations.append(json.loads(conv_data))
        
        return conversations

    async def get_conversation_history(self, limit: int = None) -> List[Dict]:
        try:
            if limit:
                entries = await self.redis_client.lrange('conversation_history', 0, limit - 1)
            else:
                entries = await self.redis_client.lrange('conversation_history', 0, -1)
            return [json.loads(entry) for entry in entries]
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []

    async def clear_conversation_history(self):
        await self.redis_client.flushdb()

    async def clear_old_conversations(self, days_to_keep: int = 30):
        cutoff_timestamp = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        old_ids = await self.redis_client.zrangebyscore(
            'conversations_by_time',
            '-inf',
            cutoff_timestamp
        )
        
        if old_ids:
            await self.redis_client.zremrangebyscore(
                'conversations_by_time',
                '-inf',
                cutoff_timestamp
            )
            
            for conv_id in old_ids:
                await self.redis_client.delete(f'chat:{conv_id}')

    async def replace_conversations(self, start_index: int, end_index: int, new_entry: Dict):
        try:
            entries = await self.get_conversation_history()
            
            updated_entries = [new_entry] + entries[end_index:]
            
            await self.redis_client.delete('conversation_history')
            
            for entry in reversed(updated_entries):
                await self.redis_client.lpush('conversation_history', json.dumps(entry))
                
            return True
        except Exception as e:
            logger.error(f"Error replacing conversations: {e}")
            return False

    async def cleanup(self):
        self.redis_client.close()
        await self.redis_client.wait_closed()
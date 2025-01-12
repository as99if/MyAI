import json
from redis.asyncio.client import Redis
from redis.commands.json.path import Path
from datetime import datetime, timedelta
import logging

class ConversationHistoryEngine:
    def __init__(self, config, redis_key: str = "conversations_history"):
        """
        Initialize the ConversationHistoryEngine with Redis connection settings.
        """
        self.config = config
        self.redis_key = redis_key
        self.redis_url = config.get('redis_host', 'localhost')
        self.redis_port = config.get('redis_port', 6379)
        self.optimized_db = config.get('redis_live_conversation_history_db', 0)
        self.main_db = config.get('redis_complete_backup_conversation_history_db', 0)
        self.redis_password = config.get('redis_password', None)
        self.client = None
        self.retry_attempts = 3
        self.retry_delay = 1
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """
        Establish a connection to the Redis server and initialize the JSON structure.
        """
        try:
            self.client = await Redis(
                host=self.redis_url,
                port=self.redis_port,
                db=self.optimized_db,
                password=self.redis_password,
                decode_responses=True
            )

            # Test connection
            await self.client.ping()

            # Enable AOF persistence
            await self.client.config_set('appendonly', 'yes')
            await self.client.config_set('appendfsync', 'everysec')

            # Initialize the JSON structure if it doesn't exist
            if not await self.client.exists(self.redis_key):
                await self.client.json().set(self.redis_key, Path.root_path(), {})

            # Trigger background save
            await self.client.bgsave()

        except Exception as e:
            raise Exception(f"Redis connection failed: {str(e)}")
    
    async def cleanup(self):
        """
        Cleanup Redis connection
        """
        # 
        if self.client:
            # await self.client.save()
            await self.client.aclose()

    async def get_all_conversations(self):
        """
        Retrieve all conversations from Redis.
        """
        return await self.client.json().get(self.redis_key)

    async def get_conversations_by_date(self, date: str):
        """
        Get conversations for a specific date.
        """
        conversations = await self.client.json().get(self.redis_key, f"$.{date}")
        if conversations[0] is None:
            return []
        return conversations[0]    # why conversations[0]? returns data as a list, data itself is a list

    async def get_todays_recent_conversation_segments(self, limit: int = 15):
        """
        Get the most recent conversation segments of today.
        """
        date = datetime.now().strftime('%d-%m-%Y')
        conversations = await self.get_conversations_by_date(date)
        if not conversations:
            return []
        return conversations[0][-limit:]

    async def get_recent_conversations_in_date_range(self, date_limit: int = 3):
        """
        Get conversations within a date range.
        """
        recent_conversations = []
        for i in range(date_limit):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            conversations = await self.get_conversations_by_date(date)
            if conversations:
                recent_conversations.extend(conversations)
        return recent_conversations

    async def add_conversation(self, messages: list):
        """
        Add new messages to today's conversation history.
        """
        date = datetime.now().strftime('%d-%m-%Y')
        await self.add_conversation_by_date(date, messages)

    async def add_conversation_by_date(self, date: str, messages: list):
        """
        Add new messages for a specific date.
        """
        timestamp = datetime.now().isoformat()
        formatted_messages = [
            {
                'role': role,
                'content': content,
                'timestamp': timestamp
            } for role, content in messages
        ]


        # Get existing conversations for the date
        existing = await self.get_conversations_by_date(date)
        
        if not existing:
            # If no conversations exist for this date, initialize with the new messages
            await self.client.json().set(self.redis_key, f"$.{date}", formatted_messages)
        else:
            # Append new messages to existing conversations
            # Extend the existing array with new messages
            for message in formatted_messages:
                print("message", message)
                await self.client.json().arrappend(self.redis_key, f"$.{date}", message)

    async def update_conversations_by_date(self, date: str, messages: list):
        """
        Update conversations for a specific date.
        """
        timestamp = datetime.now().isoformat()
        formatted_messages = [
            {
                'role': role,
                'content': content,
                'timestamp': timestamp
            } for role, content in messages
        ]

        # Replace existing conversations for the date
        await self.client.json().set(self.redis_key, f"$.{date}", formatted_messages)

    async def delete_data_by_date(self, date: str):
        """
        Delete conversations for a specific date.
        """
        await self.client.json().delete(self.redis_key, f"$.{date}")
    
    async def delete_data(self):
        """
        Delete conversations for a specific date.
        """
        await self.client.json().delete(self.redis_key)
    
    

async def run():
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_live_conversation_history_db': 4,
        'redis_password': None
    }
    
    engine = ConversationHistoryEngine(config)
    await engine.connect()
    
    # await engine.delete_data()
    
    # Get all conversations
    # all_conversations = await engine.get_all_conversations()
    # print("All conversations:", json.dumps(all_conversations, indent=2))
    # Example conversation
    messages = [
        ("human", "What is Python?"),
        ("assistant", "Python is a programming language."),
        ("human", "How do I use it?"),
        ("assistant", "You can start by installing Python..."),
        ("human", "haaaaaaah?")
    ]
    
    # Add conversation
    await engine.add_conversation(messages)
    
    # Get all conversations
    all_conversations = await engine.get_all_conversations()
    date = datetime.now().strftime('%d-%m-%Y')
    get_conversations_by_date = await engine.get_conversations_by_date(date)
    print("All conversations:", json.dumps(all_conversations, indent=2))
    
    # Cleanup
    # await engine.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())

from pathlib import Path as ospath
import json
import redis.asyncio as redis
from redis.commands.json.path import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import asyncio

class ConversationHistoryEngine:
    def __init__(self, config: Dict[str, Any], redis_key: str = "conversations_history"):
        """
        Initialize the ConversationHistoryEngine with Redis connection settings.
        
        Args:
            config (Dict[str, Any]): Redis configuration dictionary
            redis_key (str): Key for storing conversations
        """
        self.config = config
        self.redis_key = redis_key
        self.redis_url = config.get('redis_host', 'localhost')
        self.redis_port = config.get('redis_port', 6379)
        self.optimized_db = config.get('redis_live_conversation_history_db', 0)
        self.redis_password = config.get('redis_password', None)
        self.client: Optional[redis.Redis] = None
        self.retry_attempts = 3
        self.retry_delay = 1
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> None:
        """
        Establish a connection to the Redis server and initialize the JSON structure.
        
        Raises:
            Exception: If Redis connection fails
        """
        for attempt in range(self.retry_attempts):
            try:
                self.client = await redis.Redis(
                    host=self.redis_url,
                    port=self.redis_port,
                    db=self.optimized_db,
                    password=self.redis_password,
                    decode_responses=True,
                    encoding="UTF-8"
                )
                
                if await self.client.ping():
                    self.logger.info("Redis connection established successfully")
                    
                    # Enable AOF persistence
                    await self.client.config_set('appendonly', 'yes')
                    await self.client.config_set('appendfsync', 'everysec')

                    # Initialize the JSON structure if it doesn't exist
                    if not await self.client.exists(self.redis_key):
                        await self.client.json().set(self.redis_key, Path.root_path(), {})
                    return
                    
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retry_attempts - 1:
                    raise Exception(f"Redis connection failed after {self.retry_attempts} attempts: {str(e)}")
                await asyncio.sleep(self.retry_delay)

    async def cleanup(self) -> None:
        """Cleanup Redis connection"""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def _get_all_data(self) -> Any:
        """Helper method to safely get JSON data from Redis"""
        try:
            data = await self.client.json().get(self.redis_key)
            return data if data is not None else []
        except Exception as e:
            self.logger.error(f"Error getting JSON data: {str(e)}")
            return []
        
    async def _get_json_data(self, path: str) -> Any:
        """Helper method to safely get JSON data from Redis"""
        try:
            data = await self.client.json().get(self.redis_key, path)
            return data if data is not None else []
        except Exception as e:
            self.logger.error(f"Error getting JSON data for path {path}: {str(e)}")
            return []

    async def get_conversations_by_date(self, date: str) -> List[Dict[str, Any]]:
        """
        Get conversations for a specific date.
        
        Args:
            date (str): Date in format 'dd-mm-yyyy'
            
        Returns:
            List[Dict[str, Any]]: List of conversation messages
        """
        try:
            conversations = await self._get_json_data(f"$.{date}")
            if not conversations or conversations == [None]:
                return []
                
            if isinstance(conversations, list):
                return conversations[0] if len(conversations) == 1 else conversations
                
            return []
        except Exception as e:
            self.logger.error(f"Error retrieving conversations for date {date}: {str(e)}")
            return []

    async def ingest_conversation_heistory(self, conversation_history) -> None:
        await self.client.json().set(self.redis_key, Path.root_path(), conversation_history)
        
        
    async def add_conversation_by_date(self, date: str, messages: List[Dict[str, Any]]) -> None:
        """
        Add new messages for a specific date.
        
        Args:
            date (str): Date in format 'dd-mm-yyyy'
            messages (List[Dict[str, Any]]): List of message dictionaries
        """
        try:
            existing = await self._get_json_data(f"$.{date}")
            # existing = await self.client.exists(self.redis_key, f"$.{date}")
            
            if not existing:
                await self.client.json().set(self.redis_key, f"$.{date}", messages)
            else:
                existing_messages = existing[0] if isinstance(existing, list) and len(existing) == 1 else existing
                if isinstance(existing_messages, list):
                    existing_messages.extend(messages)
                    await self.client.json().set(self.redis_key, f"$.{date}", existing_messages)
                else:
                    await self.client.json().set(self.redis_key, f"$.{date}", messages)
                    
        except Exception as e:
            self.logger.error(f"Error adding conversation for date {date}: {str(e)}")
            raise

    async def add_conversation(self, messages: List[Dict[str, Any]]) -> None:
        """Add new messages to today's conversation history."""
        date = datetime.now().strftime('%d-%m-%Y')
        await self.add_conversation_by_date(date, messages)

    async def get_recent_conversations(self, days: int = 3, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Get recent conversations within a date range with limit.
        
        Args:
            days (int): Number of days to look back
            limit (int): Maximum number of messages to return
            
        Returns:
            List[Dict[str, Any]]: Limited list of recent conversations
        """
        recent_conversations = []
        for i in range(days, -1, -1):
            date = (datetime.now() - timedelta(days=i)).strftime('%d-%m-%Y')
            conversations = await self.get_conversations_by_date(date)
            # print(f"-------{date}-----")
            # print(json.dumps(conversations, indent=2))
            if conversations:
                recent_conversations.extend(conversations)
                if len(recent_conversations) >= limit:
                    return recent_conversations[:limit]
        return recent_conversations[:limit]
    
    async def delete_data_by_date(self, date: str) -> None:
        """Delete conversations for a specific date."""
        try:
            await self.client.json().delete(self.redis_key, f"$.{date}")
        except Exception as e:
            self.logger.error(f"Error deleting data for date {date}: {str(e)}")
            raise

    async def delete_all(self) -> None:
        """Delete all conversation history."""
        try:
            await self.client.json().delete(self.redis_key)
            await self.client.json().set(self.redis_key, Path.root_path(), {})
        except Exception as e:
            self.logger.error(f"Error deleting all data: {str(e)}")
            raise
"""
# Run test scenarios for the ConversationHistoryEngine.
async def tests():
    
    # This function demonstrates basic usage and functionality testing.
    
    try:
        # Test configuration
        config = {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_live_conversation_history_db': 4,
            'redis_password': None
        }

        print("\n=== Starting ConversationHistoryEngine Tests ===\n")

        # Initialize and connect
        print("1. Initializing engine and connecting to Redis...")
        engine = ConversationHistoryEngine(config)
        await engine.connect()
        print("✓ Connection established successfully")

        # Clean start
        print("\n2. Cleaning existing data...")
        await engine.delete_all()
        print("✓ Data cleaned successfully")

        sample_conversation_history_path = "/Users/asifahmed/Development/ProjectKITT/src/memory/test_conversation_history.json"
        try:
            with open(sample_conversation_history_path, "r") as f:
                sample_conversation_history = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError
        except json.JSONDecodeError:
            raise json.JSONDecodeError
        
        await engine.ingest_conversation_heistory(sample_conversation_history)
        
        
        # Test data
        date = datetime.now().strftime('%d-%m-%Y')
        
        # Test retrieving all conversation data
        print("\n3.0. Testing conversation retrieval...")
        conversations = await engine._get_all_data()
        print("Retrieved all data:")
        print(json.dumps(conversations, indent=2))
        
        messages = [
            {
                'role': 'human',
                'content': 'What is the weather like today?',
                'timestamp': (datetime.now() - timedelta(minutes=0)).isoformat()
            },
            {
                'role': 'assistant',
                'content': 'I cannot provide real-time weather information.',
                'timestamp': (datetime.now() - timedelta(minutes=1)).isoformat()
            },
            {
                'role': 'human',
                'content': 'Can you recommend a good book to read?',
                'timestamp': (datetime.now() - timedelta(minutes=2)).isoformat()
            },
            {
                'role': 'assistant',
                'content': 'Sure, how about "1984" by George Orwell? It’s a classic dystopian novel.',
                'timestamp': (datetime.now() - timedelta(minutes=3)).isoformat()
            },
            {
                'role': 'human',
                'content': 'What are some healthy snacks?',
                'timestamp': (datetime.now() - timedelta(minutes=4)).isoformat()
            },
            {
                'role': 'assistant',
                'content': 'Some healthy snacks include almonds, carrot sticks, and Greek yogurt.',
                'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat()
            }
        ]

        # Test adding conversations
        print("\n3. Testing conversation addition...")
        await engine.add_conversation(messages)
        print("✓ Conversations added successfully")
        
        

        # Test retrieving conversations
        print("\n4. Testing conversation retrieval...")
        conversations = await engine.get_conversations_by_date(date)
        print(f"Retrieved conversations by today's date {date}:")
        print(json.dumps(conversations, indent=2))

        # Test recent conversations
        print("\n5. Testing last 3 days recent conversations retrieval...")
        recent = await engine.get_recent_conversations(days=6, limit=25)
        print(f"Retrieved {len(recent)} recent messages of the last 3 days")
        print(json.dumps(recent, indent=2))
        

        # Cleanup
        print("\n6. Performing cleanup...")
        await engine.cleanup()
        print("✓ Cleanup completed")

        print("\n=== All tests completed successfully ===\n")

    except Exception as e:
        print(f"\n❌ Error during tests: {str(e)}")
        raise
    finally:
        if 'engine' in locals():
            await engine.cleanup()

if __name__ == "__main__":
    import asyncio
    
    try:
        asyncio.run(tests())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        
"""
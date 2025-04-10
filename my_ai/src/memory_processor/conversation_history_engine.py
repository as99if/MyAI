"""
ConversationHistoryEngine
------------------------
A distributed conversation history management system using Redis and local JSON storage.

Key Features:
- Dual storage system:
  * Redis DB 0: Optimized live conversation database
  * Redis DB 1: Complete backup database
  * Local JSON: File-based backup at conversation_memory.json
  
Technical Features:
- Asynchronous Redis operations with automatic reconnection
- JSON-based storage with date-wise conversation organization
- AOF persistence enabled for data durability
- Configurable time windows for conversation retrieval
- Automatic backup and synchronization between Redis and JSON

Usage Example:
    engine = ConversationHistoryEngine(config)
    await engine.connect()
    await engine.add_conversation(messages)
    conversations = await engine.get_recent_conversations(days=3)
    await engine.cleanup()
    
Testing:
    Run tests with: $ python -m conversation_history_engine

@author: Asif Ahmed <asif.shuvo2199@outlook.com>
"""

from pathlib import Path as ospath
import json
import redis.asyncio as redis
from redis.commands.json.path import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import asyncio
from src.core.api_server.data_models import MessageContent
from src.config.config import load_config


class ConversationHistoryEngine:
    def __init__(self, redis_key: str = "conversations_history"):
        """
        Initialize the ConversationHistoryEngine.
        
        Args:
            redis_key (str): Key prefix for storing conversations in Redis
            
        Note:
            Configuration is loaded from config file with the following defaults:
            - Redis host: localhost
            - Redis port: 6379
            - Live DB: 0
            - Backup DB: 1
            - Config DB: 2
            - Vector DB: 3
        """
        self.config = load_config()
        self.redis_key = redis_key
        self.config_key = "config"
        self.redis_url = self.config.get('redis_host', 'localhost')
        self.redis_port = self.config.get('redis_port', 6379)
        self.redis_password = self.config.get('redis_password', None)
        
        # for conversation
        self.conversation_db = self.config.get('redis_live_conversation_history_db', 0)
        self.backup_db = self.config.get('redis_complete_backup_conversation_history_db', 1)
        self.client: Optional[redis.Redis] = None
        self.backup_client: Optional[redis.Redis] = None
        
        # for knowledgebase and configs
        self.config_db = self.config.get('redis_config_db', 2)
        self.vector_db = self.config.get('redis_vector_db', 3)
        self.config_client: Optional[redis.Redis] = None
        self.vector_client: Optional[redis.Redis] = None
        
        self.retry_attempts = 3
        self.retry_delay = 1
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> None:
        """
        Establish connections to Redis databases and initialize storage.
        
        Performs:
        1. Connects to live, backup, vector, and config Redis DBs
        2. Enables AOF persistence for data durability
        3. Initializes JSON structures if they don't exist
        
        Raises:
            Exception: If Redis connection fails after retry attempts
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
                self.backup_client = await redis.Redis(
                    host=self.redis_url,
                    port=self.redis_port,
                    db=self.backup_db,
                    password=self.redis_password,
                    decode_responses=True,
                    encoding="UTF-8"
                )
                self.vector_client = await redis.Redis(
                    host=self.redis_url,
                    port=self.redis_port,
                    db=self.vector_db,
                    password=self.redis_password,
                    decode_responses=True,
                    encoding="UTF-8"
                )
                
                self.config_client = await redis.Redis(
                    host=self.redis_url,
                    port=self.redis_port,
                    db=self.config_db,
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
                
                if await self.backup_client.ping():
                    self.logger.info("Redis connection established successfully for the backup")
                    
                    # Enable AOF persistence
                    await self.backup_client.config_set('appendonly', 'yes')
                    await self.backup_client.config_set('appendfsync', 'everysec')

                    # Initialize the JSON structure if it doesn't exist
                    if not await self.backup_client.exists(self.redis_key):
                        await self.backup_client.json().set(self.redis_key, Path.root_path(), {})
                    
                    return
                
                if await self.vector_client.ping():
                    self.logger.info("Redis connection established successfully for vector storage")
                    
                    # Enable AOF persistence for vector storage
                    await self.vector_client.config_set('appendonly', 'yes')
                    await self.vector_client.config_set('appendfsync', 'everysec')
                    return
                
                if await self.config_client.ping():
                    self.logger.info("Redis connection established successfully for the backup")
                    
                    # Enable AOF persistence
                    await self.config_client.config_set('appendonly', 'yes')
                    await self.config_client.config_set('appendfsync', 'everysec')

                    # Initialize the JSON structure if it doesn't exist
                    if not await self.config_client.exists(self.config_key):
                        await self.config_client.json().set(self.config_key, Path.root_path(), {})
                    return
                    
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retry_attempts - 1:
                    raise Exception(f"Redis connection failed after {self.retry_attempts} attempts: {str(e)}")
                await asyncio.sleep(self.retry_delay)

    async def get_redis_url(self, client: None, db: int = 0) -> str:
        """
        Get Redis URL for the backup database.
        
        Returns:
            str: Redis URL in format 'redis://[password@]host:port/db'
        """
        if not self.client:
            raise ValueError("Backup client not initialized")
            
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_url}:{self.redis_port}/{db}"
    
    async def cleanup(self) -> None:
        """Cleanup Redis connection"""
        if self.client:
            await self.client.aclose()
            self.client = None
        if self.backup_client:
            await self.backup_client.aclose()
            self.backup_client = None
        if self.config_client:
            await self.config_client.aclose()
            self.config_client = None
        if self.vector_client:
            await self.vector_client.aclose()
            self.vector_client = None

    
    # TODO: for all the redis gets or updates, or add, do the same for my_ai/src/memory_processor/conversation_memory.json         

    async def _read_json_file(self) -> dict:
        """Read the conversation history from JSON file"""
        try:
            json_path = ospath("my_ai/src/memory_processor/conversation_memory.json")
            if json_path.exists():
                with open(json_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error reading JSON file: {str(e)}")
            return {}

    async def _write_json_file(self, data: dict) -> None:
        """Write conversation history to JSON file"""
        try:
            json_path = ospath("my_ai/src/memory_processor/conversation_memory.json")
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error writing to JSON file: {str(e)}")
        
    
    async def _get_all_data(self) -> Any:
        """Helper method to safely get JSON data from Redis"""
        try:
            data = await self.client.json().get(self.redis_key)
            backup_data = await self.backup_client.json().get(self.redis_key)
            
            return data, backup_data
        except Exception as e:
            self.logger.error(f"Error getting JSON data: {str(e)}")
            return {}, {}
        
    async def _get_json_data(self, path: str) -> Any:
        """
        Retrieve recent conversations within a specified time window.
        
        Args:
            days (int): Number of days to look back (default: 3)
            limit (int): Maximum number of messages to return (default: 15)
            
        Returns:
            List[Dict[str, Any]]: List of conversations, ordered by date (newest first)
            Limited to specified number of messages
        """
        try:
            data = await self.client.json().get(self.redis_key, path)
            backup_data = await self.backup_client.json().get(self.redis_key, path)
            return data, backup_data
        except Exception as e:
            self.logger.error(f"Error getting JSON data for path {path}: {str(e)}")
            return []

    async def get_conversations_by_date(self, date: str) -> List[MessageContent]:
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

    async def ingest_conversation_history(self, conversation_history) -> None:
        await self.client.json().set(self.redis_key, Path.root_path(), conversation_history)
        await self._write_json_file(conversation_history)
        
    async def add_conversation_by_date(self, date: str, messages:  List[MessageContent]) -> None:
        """
    Add new messages for a specific date to both Redis and JSON file.
    
    Args:
        date (str): Date in format 'dd-mm-yyyy'
        messages (List[MessageContent]): List of messages to add
    """
        try:
            # Update Redis
            existing, backup_existing = await self._get_json_data(f"$.{date}")
            if not backup_existing or backup_existing == []:
                await self.backup_client.json().set(self.redis_key, f"$.{date}", messages)
            elif not existing or existing == []:
                await self.client.json().set(self.redis_key, f"$.{date}", messages)
            else:
                existing_messages = existing[0] if isinstance(existing, list) and len(existing) == 1 else existing
                if isinstance(existing_messages, list):
                    existing_messages.extend(messages)
                await self.client.json().set(self.redis_key, f"$.{date}", existing_messages)
                await self.backup_client.json().set(self.redis_key, f"$.{date}", existing_messages)

            # Update JSON file
            json_data = await self._read_json_file()
            if date not in json_data:
                json_data[date] = messages
            else:
                if isinstance(json_data[date], list):
                    json_data[date].extend(messages)
            await self._write_json_file(json_data)

        except Exception as e:
            self.logger.error(f"Error adding conversation for date {date}: {str(e)}")
            raise
    
    async def add_conversation(self, messages: List[MessageContent]) -> None:
        """
        Add new messages to today's conversation history.
        
        Args:
            messages (List[MessageContent]): List of messages to add to today's history
        """
        date = datetime.now().strftime('%d-%m-%Y')
        await self.add_conversation_by_date(date, messages)

        

    async def get_recent_conversations(self, days: int = 3, limit: int = 15) -> List[MessageContent]:
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
    """Delete conversations for a specific date from both Redis and JSON file"""
    try:
        await self.client.json().delete(self.redis_key, f"$.{date}")
        json_data = await self._read_json_file()
        if date in json_data:
            del json_data[date]
            await self._write_json_file(json_data)
    except Exception as e:
        self.logger.error(f"Error deleting data for date {date}: {str(e)}")
        raise

async def delete_all(self) -> None:
    """Delete all conversation history from both Redis and JSON file"""
    try:
        await self.client.json().delete(self.redis_key)
        await self.client.json().set(self.redis_key, Path.root_path(), {})
        await self._write_json_file({})
    except Exception as e:
        self.logger.error(f"Error deleting all data: {str(e)}")
        raise

async def replace_conversation_history(self, conversation_history: Any) -> None:
    """Replace entire conversation history in both Redis and JSON file"""
    try:
        await self.client.json().delete(self.redis_key)
        await self.client.json().set(self.redis_key, Path.root_path(), conversation_history)
        await self._write_json_file(conversation_history)
    except Exception as e:
        self.logger.error(f"Error replacing conversation history: {str(e)}")
        raise
            
    
# Run test scenarios for the ConversationHistoryEngine.
async def tests():
    """
    Run comprehensive tests for ConversationHistoryEngine.
    
    Test Scenarios:
    1. Initialize and connect to Redis
    2. Clean existing test data
    3. Load and ingest sample conversation history
    4. Add new conversations
    5. Retrieve conversations by date
    6. Test recent conversation retrieval
    7. Cleanup and connection termination
    
    Note: Uses test data from test_conversation_history.json
    """
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
        conversations, _ = await engine._get_all_data()
        print("Retrieved all data:")
        print(json.dumps(conversations, indent=2))
        
        messages = [
            MessageContent(
                role="human",
                content="What is the weather like today?",
                timestamp=datetime.now() - timedelta(minutes=0)
            ),
            MessageContent(
                role="assistant",
                content="I cannot provide real-time weather information.",
                timestamp=datetime.now() - timedelta(minutes=1)
            ),
            MessageContent(
                role="human",
                content="Can you recommend a good book to read?",
                timestamp=datetime.now() - timedelta(minutes=2)
            ),
            MessageContent(
                role="assistant",
                content='Sure, how about "1984" by George Orwell? It's a classic dystopian novel.',
                timestamp=datetime.now() - timedelta(minutes=3)
            ),
            MessageContent(
                role="human",
                content="What are some healthy snacks?",
                timestamp=datetime.now() - timedelta(minutes=4)
            ),
            MessageContent(
                role="assistant",
                content="Some healthy snacks include almonds, carrot sticks, and Greek yogurt.",
                timestamp=datetime.now() - timedelta(minutes=5)
            )
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
    print("--- starting tests session ---")
    try:
        asyncio.run(tests())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        

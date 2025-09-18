"""
MemoryProcessor
--------------
A distributed conversation history management system using Redis and local JSON storage.

Key Features:
- Dual storage system with fallback:
  * Redis DB 0: Optimized live conversation database (primary when available)
  * Redis DB 1: Complete backup database
  * Local JSON: File-based storage that serves as both backup and fallback
  
Technical Features:
- Graceful fallback to JSON files when Redis is unavailable
- Asynchronous Redis operations with automatic reconnection
- JSON-based storage with date-wise conversation organization
- AOF persistence enabled for data durability
- Configurable time windows for conversation retrieval
- Automatic backup and synchronization between Redis and JSON

Usage Example:
    processor = MemoryProcessor()
    await processor.connect()
    await processor.add_conversation(messages)
    conversations = await processor.get_recent_conversations(days=3)
    await processor.cleanup()
    
@author: Asif Ahmed <asif.shuvo2199@outlook.com>
"""

from pathlib import Path as ospath
import json
import os
import pprint
import redis.asyncio as redis
from redis.commands.json.path import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio
from src.utils.my_ai_utils import deserialize_message_content_list, serialize_message_content_list
from src.utils.log_manager import LoggingManager
from src.core.api_server.data_models import MessageContent
from src.config.config import load_config


class MemoryProcessor:
    def __init__(self):
        """
        Initialize the MemoryProcessor.
        
        Creates a system that:
        - Uses Redis as primary storage when available
        - Falls back to JSON files when Redis is unavailable
        - Maintains both systems in sync when possible
        """
        self.config = load_config()
        
        # Redis server settings
        self.redis_url = self.config.get('redis_host', 'localhost')
        self.redis_port = self.config.get('redis_port', 6379)
        self.redis_password = self.config.get('redis_password', None)
        
        # Redis database indexes
        self.real_time_db = self.config.get('redis_real_time_db', 0)
        self.master_backup_db = self.config.get('redis_master_backup_db', 1)
        
        # Redis clients (will be initialized during connect())
        self.real_time_db_client: Optional[redis.Redis] = None
        self.master_backup_db_client: Optional[redis.Redis] = None
        
        # Redis keys for different data types
        self.redis_conversation_key = "conversation"
        self.redis_backup_conversation_key = "backup_conversation"
        self.redis_config_key = "config"
        self.redis_vector_key = "vector"

        # Connection management
        self.retry_attempts = 3
        self.retry_delay = 1
        self.redis_available = False  # Track if Redis is available
        
        # JSON file paths
        self.json_base_dir = ospath("src/memory_processor/json_storage")
        self.conversation_file = self.json_base_dir / "conversation_memory.json"
        self.config_file = self.json_base_dir / "config_storage.json"
        self.vector_file = self.json_base_dir / "vector_storage.json"
        
        # Ensure JSON storage directory exists
        os.makedirs(self.json_base_dir, exist_ok=True)
        
        # Logging
        self.logging_manager = LoggingManager()

    async def connect(self) -> None:
        """
        Establish connections to Redis databases and initialize storage.
        
        If Redis connection fails, the system will operate in JSON-only mode.
        All data operations will automatically use JSON storage in this case.
        """
        for attempt in range(self.retry_attempts):
            try:
                # Attempt to connect to real-time database
                self.real_time_db_client = await redis.Redis(
                    host=self.redis_url,
                    port=self.redis_port,
                    db=self.real_time_db,
                    password=self.redis_password,
                    decode_responses=True,
                    encoding="UTF-8"
                )
                
                # Attempt to connect to backup database
                self.master_backup_db_client = await redis.Redis(
                    host=self.redis_url,
                    port=self.redis_port,
                    db=self.master_backup_db,
                    password=self.redis_password,
                    decode_responses=True,
                    encoding="UTF-8"
                )
                
                # Test connections and initialize Redis if available
                if await self.real_time_db_client.ping() and await self.master_backup_db_client.ping():
                    self.redis_available = True
                    
                    self.logging_manager.add_message(
                        "Redis connections established successfully", 
                        level="INFO", 
                        source="MemoryProcessor"
                    )
                    
                    # Enable AOF persistence for data durability
                    await self.real_time_db_client.config_set('appendonly', 'yes')
                    await self.real_time_db_client.config_set('appendfsync', 'everysec')
                    
                    await self.master_backup_db_client.config_set('appendonly', 'yes')
                    await self.master_backup_db_client.config_set('appendfsync', 'everysec')

                    # Initialize Redis data structures if they don't exist
                    await self._initialize_redis_structures()
                    
                    # Synchronize JSON to Redis if we're transitioning from offline to online
                    await self._sync_json_to_redis()
                    
                    break  # Successfully connected
                else:
                    # Failed to ping, clean up for retry
                    await self._clean_redis_clients()
                
            except Exception as e:
                self.logging_manager.add_message(
                    f"Redis connection attempt {attempt + 1} failed: {str(e)}", 
                    level="WARNING", 
                    source="MemoryProcessor"
                )
                
                # Clean up failed connections
                await self._clean_redis_clients()
                
                # On last attempt, switch to JSON-only mode
                if attempt == self.retry_attempts - 1:
                    self.redis_available = False
                    self.logging_manager.add_message(
                        f"Operating in JSON-only mode after {self.retry_attempts} failed Redis connection attempts", 
                        level="WARNING", 
                        source="MemoryProcessor"
                    )
                    # Initialize JSON files if they don't exist
                    await self._initialize_json_files()
                else:
                    await asyncio.sleep(self.retry_delay)

    async def _initialize_redis_structures(self) -> None:
        """Initialize required Redis JSON structures if they don't exist."""
        # Initialize conversation structure in real-time DB
        if not await self.real_time_db_client.exists(self.redis_conversation_key):
            await self.real_time_db_client.json().set(self.redis_conversation_key, Path.root_path(), {})
        
        # Initialize structures in master backup DB
        if not await self.master_backup_db_client.exists(self.redis_backup_conversation_key):
            await self.master_backup_db_client.json().set(self.redis_backup_conversation_key, Path.root_path(), {})
        
        if not await self.master_backup_db_client.exists(self.redis_config_key):
            await self.master_backup_db_client.json().set(self.redis_config_key, Path.root_path(), load_config())
        
        if not await self.master_backup_db_client.exists(self.redis_vector_key):
            await self.master_backup_db_client.json().set(self.redis_vector_key, Path.root_path(), {})

    async def _initialize_json_files(self) -> None:
        """Initialize JSON files if they don't exist."""
        # Ensure conversation file exists
        if not self.conversation_file.exists():
            await self._write_json_file(self.conversation_file, {})
        
        # Ensure config file exists
        if not self.config_file.exists():
            await self._write_json_file(self.config_file, load_config())
        
        # Ensure vector file exists
        if not self.vector_file.exists():
            await self._write_json_file(self.vector_file, {})

    async def _clean_redis_clients(self) -> None:
        """Clean up Redis clients during connection retry or shutdown."""
        if self.real_time_db_client:
            await self.real_time_db_client.aclose()
            self.real_time_db_client = None
            
        if self.master_backup_db_client:
            await self.master_backup_db_client.aclose()
            self.master_backup_db_client = None

    async def _sync_json_to_redis(self) -> None:
        """
        Synchronize data from JSON to Redis when transitioning from offline to online.
        
        This ensures that any data written during JSON-only mode is preserved when
        Redis becomes available again.
        """
        try:
            # Sync conversations if JSON file exists
            if self.conversation_file.exists():
                json_data = await self._read_json_file(self.conversation_file)
                redis_data = await self.real_time_db_client.json().get(self.redis_conversation_key)
                
                # Merge JSON data into Redis
                if json_data:
                    merged_data = await self._merge_conversation_data(redis_data or {}, json_data)
                    await self.real_time_db_client.json().set(
                        self.redis_conversation_key, Path.root_path(), merged_data
                    )
                    
                    # Also update backup
                    await self.master_backup_db_client.json().set(
                        self.redis_backup_conversation_key, Path.root_path(), merged_data
                    )
                    
                    self.logging_manager.add_message(
                        "Synchronized conversation data from JSON to Redis", 
                        level="INFO", 
                        source="MemoryProcessor"
                    )
            
            # Similarly sync config and vector data
            # (Implementation similar to conversation sync)
        except Exception as e:
            self.logging_manager.add_message(
                f"Error synchronizing JSON to Redis: {str(e)}", 
                level="ERROR", 
                source="MemoryProcessor"
            )

    async def _merge_conversation_data(self, redis_data: dict, json_data: dict) -> dict:
        """
        Merge conversation data from Redis and JSON.
        
        For dates present in both sources, combines messages without duplicates.
        For dates in only one source, keeps those messages.
        
        Args:
            redis_data (dict): Conversation data from Redis
            json_data (dict): Conversation data from JSON
            
        Returns:
            dict: Merged conversation data
        """
        result = redis_data.copy()
        
        # Process each date in the JSON data
        for date, json_messages in json_data.items():
            if not json_messages:
                continue
                
            if date not in result:
                # Date only in JSON - add all messages
                result[date] = json_messages
            else:
                # Date in both - need to merge messages
                # This is simplified - a more robust implementation would avoid duplicates
                # based on message content and timestamps
                redis_messages = result[date]
                if isinstance(redis_messages, list):
                    # Create a set of message content to avoid duplicates
                    existing_contents = {msg.get('content', '') for msg in redis_messages if isinstance(msg, dict)}
                    
                    # Add only messages that don't already exist
                    for msg in json_messages:
                        if isinstance(msg, dict) and msg.get('content', '') not in existing_contents:
                            redis_messages.append(msg)
                            existing_contents.add(msg.get('content', ''))
                    
                    result[date] = redis_messages
        
        return result

    async def get_redis_url(self, db: int = 0) -> str:
        """
        Get Redis URL for the specified database.
        
        Args:
            db (int): Database number
            
        Returns:
            str: Redis URL in format 'redis://[password@]host:port/db'
        """
        if not self.redis_available:
            raise ValueError("Redis is not available")
            
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_url}:{self.redis_port}/{db}"
    
    async def cleanup(self) -> None:
        """Close Redis connections and clean up resources."""
        await self._clean_redis_clients()
        self.redis_available = False

    async def _read_json_file(self, file_path: ospath) -> dict:
        """
        Read data from a JSON file.
        
        Args:
            file_path (ospath): Path to the JSON file
            
        Returns:
            dict: Data from JSON file or empty dict if file doesn't exist
        """
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logging_manager.add_message(
                f"Error reading JSON file {file_path}: {str(e)}", 
                level="ERROR", 
                source="MemoryProcessor"
            )
            return {}

    async def _write_json_file(self, file_path: ospath, data: dict) -> None:
        """
        Write data to a JSON file.
        
        Args:
            file_path (ospath): Path to the JSON file
            data (dict): Data to write to file
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logging_manager.add_message(
                f"Error writing to JSON file {file_path}: {str(e)}", 
                level="ERROR", 
                source="MemoryProcessor"
            )

    async def _get_data_by_key(self, key: str) -> Any:
        """
        Get data for a specific key from either Redis or JSON depending on availability.
        
        Args:
            key (str): Redis key or data type identifier
            
        Returns:
            Any: Data for the specified key
        """
        if self.redis_available:
            try:
                # Get from appropriate Redis database based on key
                if key == self.redis_conversation_key:
                    return await self.real_time_db_client.json().get(key)
                elif key in [self.redis_backup_conversation_key, self.redis_config_key, self.redis_vector_key]:
                    return await self.master_backup_db_client.json().get(key)
            except Exception as e:
                self.logging_manager.add_message(
                    f"Redis error getting data for key {key}: {str(e)}", 
                    level="ERROR", 
                    source="MemoryProcessor"
                )
                # Fall back to JSON if Redis operation fails
                self.redis_available = False
        
        # If Redis is not available or operation failed, use JSON
        if key == self.redis_conversation_key or key == self.redis_backup_conversation_key:
            return await self._read_json_file(self.conversation_file)
        elif key == self.redis_config_key:
            return await self._read_json_file(self.config_file)
        elif key == self.redis_vector_key:
            return await self._read_json_file(self.vector_file)
        
        return {}

    async def _get_data_by_path(self, key: str, path: str) -> Any:
        """
        Get data from a specific path in a JSON structure from Redis or JSON file.
        
        Args:
            key (str): Redis key or data type identifier
            path (str): JSON path to retrieve
            
        Returns:
            Any: Data at the specified path
        """
        if self.redis_available:
            try:
                # Get from appropriate Redis database based on key
                if key == self.redis_conversation_key:
                    return await self.real_time_db_client.json().get(key, path)
                elif key in [self.redis_backup_conversation_key, self.redis_config_key, self.redis_vector_key]:
                    return await self.master_backup_db_client.json().get(key, path)
            except Exception as e:
                self.logging_manager.add_message(
                    f"Redis error getting data at path {path} for key {key}: {str(e)}", 
                    level="ERROR", 
                    source="MemoryProcessor"
                )
                # Fall back to JSON if Redis operation fails
                self.redis_available = False
        
        # Extract data from path manually from JSON if Redis not available
        # This is a simplified version for the $.date pattern
        try:
            file_path = self._get_file_path_for_key(key)
            data = await self._read_json_file(file_path)
            
            # Handle simple path format "$.date"
            if path.startswith("$."):
                path_parts = path[2:].split('.')
                result = data
                for part in path_parts:
                    if isinstance(result, dict) and part in result:
                        result = result[part]
                    else:
                        return []
                return result
            
            return []
        except Exception as e:
            self.logging_manager.add_message(
                f"Error extracting data from JSON at path {path}: {str(e)}", 
                level="ERROR", 
                source="MemoryProcessor"
            )
            return []

    def _get_file_path_for_key(self, key: str) -> ospath:
        """Get the appropriate JSON file path for a given key."""
        if key == self.redis_conversation_key or key == self.redis_backup_conversation_key:
            return self.conversation_file
        elif key == self.redis_config_key:
            return self.config_file
        elif key == self.redis_vector_key:
            return self.vector_file
        else:
            return self.conversation_file  # Default fallback

    async def _set_data_by_path(self, key: str, path: str, data: Any) -> None:
        """
        Set data at a specific path in Redis and/or JSON file.
        
        Args:
            key (str): Redis key or data type identifier
            path (str): JSON path to set
            data (Any): Data to set at the specified path
        """
        # Always update JSON files regardless of Redis availability
        try:
            # print("\n****_set_data_by_path. data:***")
            # pprint.pprint(data)
            # print("\n")
            

            file_path = self._get_file_path_for_key(key)
            # print(f"\n**write to json file*\n{file_path}")
            # prints "src/memory_processor/json_storage/conversation_memory.json"
            json_data = await self._read_json_file(file_path)
            
            # Handle simple path format "$.date"
            if path == Path.root_path():
                # Replace entire structure
                json_data = data
            elif path.startswith("$."):
                # Set specific path
                target_key = path[2:]  # Remove ".$" prefix
                json_data[target_key] = data
            
            # print(f"\n*json data**\n{json_data}")
            
            await self._write_json_file(file_path, json_data)
        except Exception as e:
            self.logging_manager.add_message(
                f"Error setting data in JSON at path {path}: {str(e)}", 
                level="ERROR", 
                source="MemoryProcessor"
            )
        
        # Update Redis if available
        if self.redis_available:
            try:
                if key == self.redis_conversation_key:
                    await self.real_time_db_client.json().set(key, path, data)
                    # Also update backup
                    if path == Path.root_path() or path.startswith("$."):
                        await self.master_backup_db_client.json().set(
                            self.redis_backup_conversation_key, path, data
                        )
                elif key == self.redis_backup_conversation_key:
                    await self.master_backup_db_client.json().set(key, path, data)
                elif key == self.redis_config_key:
                    await self.master_backup_db_client.json().set(key, path, data)
                elif key == self.redis_vector_key:
                    await self.master_backup_db_client.json().set(key, path, data)
            except Exception as e:
                self.logging_manager.add_message(
                    f"Redis error setting data at path {path} for key {key}: {str(e)}", 
                    level="ERROR", 
                    source="MemoryProcessor"
                )
                self.redis_available = False

    async def _delete_data_by_path(self, key: str, path: str) -> None:
        """
        Delete data at a specific path from Redis and JSON file.
        
        Args:
            key (str): Redis key or data type identifier
            path (str): JSON path to delete
        """
        # Update JSON files
        try:
            file_path = self._get_file_path_for_key(key)
            json_data = await self._read_json_file(file_path)
            
            if path == Path.root_path():
                # Replace with empty dict
                json_data = {}
            elif path.startswith("$."):
                # Delete specific path
                target_key = path[2:]  # Remove ".$" prefix
                if target_key in json_data:
                    del json_data[target_key]
            
            await self._write_json_file(file_path, json_data)
        except Exception as e:
            self.logging_manager.add_message(
                f"Error deleting data from JSON at path {path}: {str(e)}", 
                level="ERROR", 
                source="MemoryProcessor"
            )
        
        # Update Redis if available
        if self.redis_available:
            try:
                if key == self.redis_conversation_key:
                    await self.real_time_db_client.json().delete(key, path)
                    # Also update backup
                    if path.startswith("$."):
                        await self.master_backup_db_client.json().delete(
                            self.redis_backup_conversation_key, path
                        )
                elif key == self.redis_backup_conversation_key:
                    await self.master_backup_db_client.json().delete(key, path)
                elif key == self.redis_config_key:
                    await self.master_backup_db_client.json().delete(key, path)
                elif key == self.redis_vector_key:
                    await self.master_backup_db_client.json().delete(key, path)
            except Exception as e:
                self.logging_manager.add_message(
                    f"Redis error deleting data at path {path} for key {key}: {str(e)}", 
                    level="ERROR", 
                    source="MemoryProcessor"
                )
                self.redis_available = False

    
    
    async def set_config_param(self, config_arg: str, config_param: Any) -> None:
        """
        Update a configuration parameter in storage.
        
        Args:
            config_arg (str): Configuration parameter name
            config_param (Any): New value for the configuration parameter
        """
        await self._set_data_by_path(self.redis_config_key, f"$.{config_arg}", config_param)
        self.logging_manager.add_message(
            f"Config param changed: {config_arg} = {str(config_param)}", 
            level="INFO", 
            source="MemoryProcessor"
        )

    async def get_conversations_by_date(self, date: str) -> List[MessageContent]:
        """
        Retrieve conversations for a specific date.
        
        Args:
            date (str): Date in format 'dd-mm-yyyy'
            
        Returns:
            List[MessageContent]: Conversation messages for the specified date
        """
        conversations = await self._get_data_by_path(self.redis_conversation_key, f"$.{date}")
        
        if not conversations or conversations == [None]:
            return []
            
        if isinstance(conversations, list):
            # convert json to [MessageContent] here
            conversations = deserialize_message_content_list(conversations)
            return conversations
        
        return []

    async def ingest_conversation_history(self, conversation_history: dict) -> None:
        """
        Import a complete conversation history, replacing existing data.
        
        Args:
            conversation_history (dict): Complete conversation history to import
        """
        conversation_history = serialize_message_content_list(conversation_history)
        await self._set_data_by_path(self.redis_conversation_key, Path.root_path(), conversation_history)
        self.logging_manager.add_message(
            "Ingested new conversation history", 
            level="INFO", 
            source="MemoryProcessor"
        )

    async def add_conversation_by_date(self, date: str, messages: List[MessageContent]) -> None:
        """
        Add new messages for a specific date.
        
        Args:
            date (str): Date in format 'dd-mm-yyyy'
            messages (List[MessageContent]): List of messages to add
        """
        try:
            # Get existing messages for the date
            self.logging_manager.add_message(f"Adding to conversation memory for {date}", level="INFO", source="MemoryProcessor")
            existing_messages = await self.get_conversations_by_date(date)
            # print("\n\n***existing_messages***\n\n")
            # print(existing_messages)
            # Combine with new messages
            if not existing_messages:
                combined_messages = messages
            else:
                if isinstance(existing_messages, list):
                    existing_messages.extend(messages)
                    combined_messages = existing_messages
                else:
                    combined_messages = messages
            # print("\n\ncombined_messages\n\n")
            # print(combined_messages)
            # Store updated messages

            # convert [MessageContent] into json here
            combined_messages = serialize_message_content_list(combined_messages)
            await self._set_data_by_path(self.redis_conversation_key, f"$.{date}", combined_messages)
            
        except Exception as e:
            self.logging_manager.add_message(
                f"Error adding conversation for date {date}: {str(e)}", 
                level="ERROR", 
                source="MemoryProcessor"
            )
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
        Get recent conversations within a specified time window, limited by count.
        
        Args:
            days (int): Number of days to look back
            limit (int): Maximum number of messages to return
            
        Returns:
            List[MessageContent]: Limited list of recent conversations
        """
        recent_conversations = []
        
        # Iterate through dates from oldest to newest (for chronological order)
        for i in range(days, -1, -1):
            date = (datetime.now() - timedelta(days=i)).strftime('%d-%m-%Y')
            conversations = await self.get_conversations_by_date(date)
            # make it [MessageContent]
            conversations = deserialize_message_content_list(conversations)
            if conversations:
                recent_conversations.extend(conversations)
                # Return if we've reached the limit
                if len(recent_conversations) >= limit:
                    return recent_conversations[:limit]
        
        recent_conversations = recent_conversations[:limit]
        return recent_conversations

    async def delete_data_by_date(self, date: str) -> None:
        """
        Delete conversations for a specific date.
        
        Args:
            date (str): Date in format 'dd-mm-yyyy' to delete
        """
        await self._delete_data_by_path(self.redis_conversation_key, f"$.{date}")
        self.logging_manager.add_message(
            f"Deleted conversation data for date {date}", 
            level="INFO", 
            source="MemoryProcessor"
        )

    async def delete_all(self) -> None:
        """Delete all conversation history from storage."""
        # Set empty dict for conversations
        await self._set_data_by_path(self.redis_conversation_key, Path.root_path(), {})
        self.logging_manager.add_message(
            "Deleted all conversation data", 
            level="INFO", 
            source="MemoryProcessor"
        )

    async def replace_conversation_history(self, conversation_history: dict) -> None:
        """
        Replace entire conversation history.
        
        Args:
            conversation_history (dict): New conversation history to replace existing data
        """
        # Delete existing data and set new data
        # convert [MessageContent] into json here
        conversation_history = serialize_message_content_list(conversation_history)


        await self._set_data_by_path(self.redis_conversation_key, Path.root_path(), conversation_history)
        self.logging_manager.add_message(
            "Replaced conversation history", 
            level="INFO", 
            source="MemoryProcessor"
        )


# Test function for the MemoryProcessor
async def tests():
    """
    Run comprehensive tests for MemoryProcessor.
    
    Test Scenarios:
    1. Initialize and connect to Redis (or fallback to JSON)
    2. Clean existing test data
    3. Load and ingest sample conversation history
    4. Add new conversations
    5. Retrieve conversations by date
    6. Test recent conversation retrieval
    7. Cleanup and connection termination
    """
    try:
        print("\n=== Starting MemoryProcessor Tests ===\n")

        # Initialize and connect
        print("1. Initializing processor and connecting to Redis/JSON...")
        processor = MemoryProcessor()
        await processor.connect()
        print(f"✓ Connection established successfully (Redis available: {processor.redis_available})")

        # Clean start
        print("\n2. Cleaning existing data...")
        await processor.delete_all()
        print("✓ Data cleaned successfully")

        # Load sample test data from a path or create synthetic test data
        sample_conversation_history = {
            "01-01-2023": [
                {
                    "role": "human",
                    "content": "Hello, this is a test message.",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "role": "assistant",
                    "content": "This is a test response.",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        # Ingest test data
        print("\n3. Ingesting sample conversation history...")
        await processor.ingest_conversation_history(sample_conversation_history)
        print("✓ Sample history ingested successfully")
        
        # Get today's date for testing
        date = datetime.now().strftime('%d-%m-%Y')
        
        # Create test messages
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
            )
        ]

        # Test adding conversations
        print("\n4. Testing conversation addition...")
        await processor.add_conversation(messages)
        print("✓ Conversations added successfully")

        # Test retrieving conversations
        print("\n5. Testing conversation retrieval...")
        conversations = await processor.get_conversations_by_date(date)
        print(f"Retrieved conversations for today ({date}):")
        print(json.dumps(conversations, indent=2))

        # Test recent conversations
        print("\n6. Testing recent conversations retrieval...")
        recent = await processor.get_recent_conversations(days=6, limit=10)
        print(f"Retrieved {len(recent)} recent messages")
        print(json.dumps(recent, indent=2))

        # Cleanup
        print("\n7. Performing cleanup...")
        await processor.cleanup()
        print("✓ Cleanup completed")

        print("\n=== All tests completed successfully ===\n")

    except Exception as e:
        print(f"\n❌ Error during tests: {str(e)}")
        raise
    finally:
        if 'processor' in locals():
            await processor.cleanup()

if __name__ == "__main__":
    import asyncio
    print("--- starting tests session ---")
    try:
        asyncio.run(tests())
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        

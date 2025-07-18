
import asyncio
from datetime import datetime, timedelta
import os
from typing import Any, List, Optional
from redis.commands.json.path import Path
from pathlib import Path as ospath
import redis.asyncio as redis

from my_ai_mcp_server.src.utils.schemas import AgentTask
from src.config.config import load_config
from src.utils.log_manager import LoggingManager

class AgentMemoryProcessor:

    def __init__(self):
        self.logging_manager = LoggingManager()

        self.config = load_config()
        
        # Redis server settings
        self.redis_url = self.config.get('redis_host', 'localhost')
        self.redis_port = self.config.get('redis_port', 6379)
        self.redis_password = self.config.get('redis_password', None)
        
        # Redis database indexes
        self.master_backup_db = self.config.get('redis_master_backup_db', 1)
        
        # Redis clients (will be initialized during connect())
        self.master_backup_db_client: Optional[redis.Redis] = None
        
        # Redis keys for different data types
        self.redis_config_key = "config"
        self.redis_agent_tasks_key = "agent_tasks"

        # Connection management
        self.retry_attempts = 3
        self.retry_delay = 1
        self.redis_available = False  # Track if Redis is available
        
        # JSON file paths
        self.json_base_dir = ospath("src/memory_processor/json_storage")
        self.agent_tasks_file = self.json_base_dir / "agent_task_memory.json"
        
        # Ensure JSON storage directory exists
        os.makedirs(self.json_base_dir, exist_ok=True)


    async def connect(self) -> None:
        """
        Establish connections to Redis databases and initialize storage.
        
        If Redis connection fails, the system will operate in JSON-only mode.
        All data operations will automatically use JSON storage in this case.
        """
        for attempt in range(self.retry_attempts):
            try:
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
                if await self.master_backup_db_client.ping():
                    self.redis_available = True
                    
                    self.logging_manager.add_message(
                        "Redis connections established successfully", 
                        level="INFO", 
                        source="MemoryProcessor"
                    )
                    
                    # Enable AOF persistence for data durability
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
        
        if not await self.master_backup_db_client.exists(self.redis_vector_key):
            await self.master_backup_db_client.json().set(self.redis_vector_key, Path.root_path(), {})

    async def _initialize_json_files(self) -> None:
        """Initialize JSON files if they don't exist."""
        # Ensure conversation file exists
        if not self.agent_tasks_file.exists():
            await self._write_json_file(self.agent_tasks_file, {})
        
    async def _clean_redis_clients(self) -> None:
        """Clean up Redis clients during connection retry or shutdown."""
            
        if self.master_backup_db_client:
            await self.master_backup_db_client.aclose()
            self.master_backup_db_client = None


    async def get_agent_tasks_by_date(self, date: str) -> List[AgentTask]:
        """
        Retrieve agent tasks for a specific date.
        
        Args:
            date (str): Date in format 'dd-mm-yyyy'
            
        Returns:
            List[dict]: Agent task records for the specified date
        """
        tasks = await self._get_data_by_path(self.redis_agent_tasks_key, f"$.{date}")
        
        if not tasks or tasks == [None]:
            return []
            
        if isinstance(tasks, list):
            # TODO: deserialize here [AgentTask]
            return tasks
        
        return []

    def _get_file_path_for_key(self, key: str) -> ospath:
        """Get the appropriate JSON file path for a given key."""
        if key == self.redis_agent_tasks_key:
            return self.conversation_file

        
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
                if key == self.redis_agent_tasks_key:
                    data = await self.master_backup_db_client.json().get(key, path)
                    return data
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
                if key == self.redis_agent_tasks_key:
                    if path == Path.root_path() or path.startswith("$."):
                        await self.master_backup_db_client.json().set(
                            self.redis_backup_conversation_key, path, data
                        )
                else:
                    self.logging_manager.add_message(
                    f"Redis key error - key {key}", 
                    level="ERROR", 
                    source="MemoryProcessor"
                )
            except Exception as e:
                self.logging_manager.add_message(
                    f"Redis error setting data at path {path} for key {key}: {str(e)}", 
                    level="ERROR", 
                    source="MemoryProcessor"
                )
                self.redis_available = False


    async def add_agent_tasks_by_date(self, date: str, tasks: List[dict]) -> None:
        """
        Add new agent task records for a specific date.
        
        Args:
            date (str): Date in format 'dd-mm-yyyy'
            tasks (List[dict]): List of task records to add
        """
        try:
            # Get existing tasks for the date
            existing_tasks = await self.get_agent_tasks_by_date(date)
            
            # Combine with new tasks
            if not existing_tasks:
                combined_tasks = tasks
            else:
                if isinstance(existing_tasks, list):
                    existing_tasks.extend(tasks)
                    combined_tasks = existing_tasks
                else:
                    combined_tasks = tasks
            
            # Store updated tasks
            # TODO: serialize here
            await self._set_data_by_path(self.redis_agent_tasks_key, f"$.{date}", combined_tasks)
            
        except Exception as e:
            self.logging_manager.add_message(
                f"Error adding agent tasks for date {date}: {str(e)}", 
                level="ERROR", 
                source="MemoryProcessor"
            )
            raise

    async def add_agent_tasks(self, tasks: List[dict]) -> None:
        """
        Add new agent task records to today's history.
        
        Args:
            tasks (List[dict]): List of task records to add to today's history
        """
        date = datetime.now().strftime('%d-%m-%Y')
        await self.add_agent_tasks_by_date(date, tasks)


    async def delete_agent_tasks_by_date(self, date: str) -> None:
        """
        Delete agent tasks for a specific date.
        
        Args:
            date (str): Date in format 'dd-mm-yyyy' to delete
        """
        await self._delete_data_by_path(self.redis_agent_tasks_key, f"$.{date}")
        self.logging_manager.add_message(
            f"Deleted agent task data for date {date}", 
            level="INFO", 
            source="MemoryProcessor"
        )

    async def delete_all_agent_tasks(self) -> None:
        """Delete all agent task history from storage."""
        await self._set_data_by_path(self.redis_agent_tasks_key, Path.root_path(), {})
        self.logging_manager.add_message(
            "Deleted all agent task data", 
            level="INFO", 
            source="MemoryProcessor"
        )
from pathlib import Path as ospath
import json
import redis.asyncio as redis
from redis.commands.json.path import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import asyncio


class AppConfiguration:
    def __init__(self, config: Dict[str, Any], redis_key: str = "config"):
        """
        Initialize the AppConfiguration class with Redis connection settings.
        
        Args:
            config (Dict[str, Any]): Redis configuration dictionary
            redis_key (str): Key for storing conversations
        """
        self.config = config
        self.config_key = redis_key
        self.redis_url = config.get('redis_host', 'localhost')
        self.redis_port = config.get('redis_port', 6379)
        self.config_db = config.get('redis_config_db', 2)
        self.redis_password = config.get('redis_password', None)
        self.config_client: Optional[redis.Redis] = None
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
                self.config_client = await redis.Redis(
                    host=self.redis_url,
                    port=self.redis_port,
                    db=self.config_db,
                    password=self.redis_password,
                    decode_responses=True,
                    encoding="UTF-8"
                )
                
                if await self.config_client.ping():
                    self.logger.info("Redis connection established successfully for the backup")
                    
                    # Enable AOF persistence
                    await self.config_client.config_set('appendonly', 'yes')
                    await self.config_client.config_set('appendfsync', 'everysec')

                    # Initialize the JSON structure if it doesn't exist
                    if not await self.config_client.exists(self.config_key):
                        await self.config_client.json().set(self.config_key, Path.root_path(), {})
                    if not await self.config_client.exists(self.config_key):
                        await self.config_client.json().set(self.config_key, Path.root_path(), {})
                    return
                    
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retry_attempts - 1:
                    raise Exception(f"Redis connection failed after {self.retry_attempts} attempts: {str(e)}")
                await asyncio.sleep(self.retry_delay)

    async def cleanup(self) -> None:
        """Cleanup Redis connection"""
        if self.config_client:
            await self.config_client.aclose()
            self.config_client = None

    
            
    async def _get_all_data(self) -> Any:
        """Helper method to safely get JSON data from Redis"""
        try:
            config = await self.config_client.json().get(self.redis_key)
            
            return config
        except Exception as e:
            self.logger.error(f"Error getting JSON data: {str(e)}")
            raise Exception(f"Error getting JSON data: {str(e)}")
            return None
        
    async def _get_json_data(self, path: str) -> Any:
        """Helper method to safely get JSON data from Redis"""
        try:
            data = await self.config.json().get(self.redis_key, path)
            return data
        except Exception as e:
            self.logger.error(f"Error getting JSON data for path {path}: {str(e)}")
            raise Exception(f"Error getting JSON data for path {path}: {str(e)}")
            return None
import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from src.memory.conversation_hostory_service import ConversationHistoryEngine


class MockRedis:
    def __init__(self):
        self.data = {}
    
    async def hset(self, *args):
        return True
    
    async def hget(self, key, field):
        return json.dumps(self.data.get(key, {}))
    
    async def zadd(self, *args):
        return True
    
    async def expire(self, *args):
        return True
    
    async def close(self):
        pass
    
    async def wait_closed(self):
        pass

@pytest.mark.asyncio
async def mock_redis():
    return MockRedis()

@pytest.mark.asyncio
async def test_config():
    return {
        "redis_host": "localhost",
        "redis_port": 6379,
        "max_context_messages": 5
    }

@pytest.mark.asyncio
async def conversation_service(mock_redis, test_config):
    service = ConversationHistoryEngine(test_config)
    service.redis_client = mock_redis
    return service

@pytest.mark.asyncio
async def test_basic_conversation(conversation_service):
    test_messages = [
        ("Who are you?", "I am an AI assistant."),
        ("What's your purpose?", "I help users with various tasks."),
        ("Remember this: sky is blue", "I'll remember that the sky is blue."),
        ("What did I ask you to remember?", "You asked me to remember that the sky is blue."),
        ("Summarize our conversation", "We discussed my identity, purpose, and talked about the sky being blue.")
    ]
    
    for human_msg, ai_msg in test_messages:
        await conversation_service.save_chat_segment([
            ('human', human_msg),
            ('assistant', ai_msg)
        ])
        
        # Verify storage
        history = await conversation_service.get_conversation_history(1)
        assert len(history) > 0

@pytest.mark.asyncio
async def test_conversation_retrieval(conversation_service):
    # Save test conversation
    await conversation_service.save_chat_segment([
        ('human', 'Test message'),
        ('assistant', 'Test response')
    ])
    
    # Retrieve and verify
    history = await conversation_service.get_conversation_history()
    assert len(history) > 0
    assert isinstance(history[0], dict)

@pytest.mark.asyncio
async def test_conversation_cleanup(conversation_service):
    # Save test data
    await conversation_service.save_chat_segment([
        ('human', 'Test'),
        ('assistant', 'Response')
    ])
    
    # Clear and verify
    await conversation_service.clear_conversation_history()
    history = await conversation_service.get_conversation_history()
    assert len(history) == 0

if __name__ == "__main__":
    pytest.main(["-v", "test_conversation_history_service.py"])
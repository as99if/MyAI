import pytest
from datetime import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.conversation_history_service import ConversationHistoryEngine

@pytest.fixture
def config():
    return {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_live_conversation_history_db': 0,
        'redis_password': None
    }

@pytest.fixture
async def engine(config):
    engine = ConversationHistoryEngine(config)
    with patch('redis.asyncio.client.Redis') as mock_redis:
        # Setup mock Redis client
        engine.client = AsyncMock()
        engine.client.ping = AsyncMock()
        engine.client.config_set = AsyncMock()
        engine.client.exists = AsyncMock(return_value=False)
        engine.client.json = MagicMock()
        engine.client.json().set = AsyncMock()
        engine.client.json().get = AsyncMock()
        engine.client.json().arrappend = AsyncMock()
        engine.client.json().delete = AsyncMock()
        engine.client.bgsave = AsyncMock()
        yield engine
        await engine.cleanup()

@pytest.mark.asyncio
async def test_initialization(config):
    engine = ConversationHistoryEngine(config)
    assert engine.redis_url == 'localhost'
    assert engine.redis_port == 6379
    assert engine.db == 0
    assert engine.redis_password is None


@pytest.mark.asyncio
async def test_add_and_get_conversation(engine):
    await engine.connect()
    
    messages = [
        ("human", "Hello"),
        ("assistant", "Hi there")
    ]
    
    date = datetime.now().strftime('%d-%m-%Y')
    await engine.add_conversation(messages)

    # Mock the return value for get_conversations_by_date
    expected_messages = [{
        'role': 'human',
        'content': 'Hello',
        'timestamp': datetime.now().isoformat()
    }, {
        'role': 'assistant',
        'content': 'Hi there',
        'timestamp': datetime.now().isoformat()
    }]
    # engine.client.json().get.return_value = expected_messages
    
    result = await engine.get_conversations_by_date(date)
    print(result)
    assert len(result) == 2
    assert result[0]['role'] == 'human'
    assert result[1]['role'] == 'assistant'

@pytest.mark.asyncio
async def test_get_recent_conversation_segments(engine):
    await engine.connect()
    
    # Mock return value
    mock_conversations = [{'role': 'human', 'content': 'test'}]
    engine.client.json().get.return_value = mock_conversations
    
    result = await engine.get_recent_conversation_segments(limit=1)
    assert len(result) == 1

@pytest.mark.asyncio
async def test_delete_data_by_date(engine):
    await engine.connect()
    date = datetime.now().strftime('%d-%m-%Y')
    await engine.delete_data_by_date(date)
    engine.client.json().delete.assert_called_once()

@pytest.mark.asyncio
async def test_cleanup(engine):
    await engine.connect()
    await engine.cleanup()
    engine.client.aclose.assert_called_once()

@pytest.mark.asyncio
async def test_get_recent_conversations_in_date_range(engine):
    await engine.connect()
    
    # Mock return value
    mock_conversations = [{'role': 'human', 'content': 'test'}]
    engine.client.json().get.return_value = mock_conversations
    
    result = await engine.get_recent_conversations_in_date_range(date_limit=1)
    assert len(result) == 1

@pytest.mark.asyncio
async def test_update_conversations_by_date(engine):
    await engine.connect()
    date = datetime.now().strftime('%d-%m-%Y')
    messages = [("human", "Updated message")]
    
    await engine.update_conversations_by_date(date, messages)
    engine.client.json().set.assert_called()

@pytest.mark.asyncio
async def test_redis_connection_failure():
    config = {'redis_host': 'invalid_host'}
    engine = ConversationHistoryEngine(config)
    
    with pytest.raises(Exception) as exc_info:
        await engine.connect()
    assert "Redis connection failed" in str(exc_info.value)
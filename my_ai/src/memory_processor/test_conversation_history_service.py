import json
from pathlib import Path
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from .conversation_history_engine import ConversationHistoryEngine

@pytest.fixture
def config():
    return {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_live_conversation_history_db': 15,  # Using DB 15 for testing
        'redis_password': None
    }

@pytest_asyncio.fixture
async def engine(config):
    engine = ConversationHistoryEngine(config)
    await engine.connect()
    yield engine
    await engine.delete_all()
    await engine.cleanup()


@pytest.fixture
def sample_messages():
    current_time = datetime.now()
    return [
        {
            'role': 'human',
            'content': 'Hello',
            'timestamp': current_time.isoformat()
        },
        {
            'role': 'assistant',
            'content': 'Hi there',
            'timestamp': (current_time + timedelta(seconds=1)).isoformat()
        }
    ]

@pytest.mark.asyncio
async def test_connection(engine):
    assert engine.client is not None
    assert await engine.client.ping() is True

@pytest.mark.asyncio
async def test_add_and_get_conversation(engine, sample_messages):
    await engine.add_conversation(sample_messages)
    date = datetime.now().strftime('%d-%m-%Y')
    conversations = await engine.get_conversations_by_date(date)
    
    assert len(conversations) == 2
    assert conversations[0]['role'] == 'human'
    assert conversations[0]['content'] == 'Hello'
    assert conversations[1]['role'] == 'assistant'
    assert conversations[1]['content'] == 'Hi there'

@pytest.mark.asyncio
async def test_get_recent_conversations(engine, sample_messages):
    # Add conversations for multiple days
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime('%d-%m-%Y')
    
    await engine.add_conversation(sample_messages)
    await engine.add_conversation_by_date(yesterday, sample_messages)
    
    # Test with different limits and days
    recent_3_days = await engine.get_recent_conversations(days=3, limit=5)
    recent_1_day = await engine.get_recent_conversations(days=1, limit=2)
    
    assert len(recent_3_days) == 4  # 2 messages each for today and yesterday
    assert len(recent_1_day) == 2   # Limited to 2 messages from today

@pytest.mark.asyncio
async def test_delete_operations(engine, sample_messages):
    date = datetime.now().strftime('%d-%m-%Y')
    
    # Test delete by date
    await engine.add_conversation(sample_messages)
    await engine.delete_data_by_date(date)
    conversations = await engine.get_conversations_by_date(date)
    assert conversations == []
    
    # Test delete all
    await engine.add_conversation(sample_messages)
    await engine.delete_all()
    conversations = await engine.get_conversations_by_date(date)
    assert conversations == []

@pytest.mark.asyncio
async def test_error_handling(config):
    invalid_config = {**config, 'redis_host': 'invalid_host'}
    engine = ConversationHistoryEngine(invalid_config)
    
    with pytest.raises(Exception) as exc_info:
        await engine.connect()
    assert "Redis connection failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_add_conversation_by_date(engine, sample_messages):
    date = datetime.now().strftime('%d-%m-%Y')
    
    # Test adding new conversation
    await engine.add_conversation_by_date(date, sample_messages[:1])
    conversations = await engine.get_conversations_by_date(date)
    assert len(conversations) == 1
    
    # Test appending to existing conversation
    await engine.add_conversation_by_date(date, sample_messages[1:])
    conversations = await engine.get_conversations_by_date(date)
    assert len(conversations) == 2

@pytest.mark.asyncio
async def test_get_nonexistent_date(engine):
    date = "01-01-2000"
    conversations = await engine.get_conversations_by_date(date)
    assert conversations == []
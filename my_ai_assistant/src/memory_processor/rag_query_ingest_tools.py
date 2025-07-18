from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import os
from dotenv import load_dotenv


def load_documents(data_dir):
    """Load documents from a directory"""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    # or use markitdown
    return documents


def process_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    return texts


def create_vector_store(redis_url, index_name, texts):
    """Create Redis vector store from processed documents"""
    embeddings = OpenAIEmbeddings()
    # get embedding of the used model

    vector_store = Redis.from_documents(
        documents=texts,
        embedding=embeddings,
        redis_url=redis_url,
        index_name=index_name
    )
    return vector_store


def ingest_data(data_dir):
    """Main method to ingest data into Redis vector store"""
    documents = load_documents(data_dir)
    texts = process_documents(documents)
    vector_store = create_vector_store(texts)
    return vector_store


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


def initialize_vector_store_from_existing(redis_url, index_name):
    """Initialize Redis vector store"""
    embeddings = OpenAIEmbeddings()
    vector_store = Redis.from_existing_index(
        embedding=embeddings,
        redis_url=redis_url,
        index_name=index_name
    )
    return vector_store


def create_rag_chain(db_key: str = "backup_db", inference_client = None):
    # https://github.com/langchain-ai/rag-from-scratch
    # https://github.com/NisaarAgharia/Advanced_RAG
    # https://github.com/NirDiamant/RAG_Techniques
    """Create RAG chain with conversation memory"""
    vector_store = initialize_vector_store_from_existing()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    memory = ConversationBufferMemory(
        memory_key=db_key,
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=inference_client,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return chain


def rag_query(question, inference_client, db_key: str = "backup_db"):
    """Query the RAG system"""
    chain = create_rag_chain(inference_client)
    response = chain({"question": question})
    return {
        "answer": response["answer"],
        "source_documents": response["source_documents"]
    }

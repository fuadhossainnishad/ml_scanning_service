import os
from dotenv import load_dotenv

load_dotenv()


config = {
    "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
    "pinecone_region": os.getenv("PINECONE_REGION", "us-east-1"),
    "embedding_dim": 2048  # depends on your embedding model
}

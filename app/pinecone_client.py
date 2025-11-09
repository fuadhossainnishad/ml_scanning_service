# app/pinecone_client.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "learning"
EMBEDDING_DIM = 512
CLOUD = "aws"
REGION = "us-east-1"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = None  # will hold the connected index

def init_index():
    global index
    existing_indexes_raw = pc.list_indexes()
    existing_indexes = []

    for idx in existing_indexes_raw:
        if hasattr(idx, "get"):  # dict-like
            existing_indexes.append(idx.get("name"))
        elif isinstance(idx, str):
            existing_indexes.append(idx)

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION)
        )
        print(f"Index '{INDEX_NAME}' created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists, skipping creation.")

    index = pc.Index(INDEX_NAME)
    print(f"Connected to Pinecone index '{INDEX_NAME}'")
    return index


# Upsert example
def upsert_vector(vector_id: str, product_id: str, embedding: list, category: str):
    if index is None:
        raise RuntimeError("Pinecone index not initialized. Call init_index() first.")

    try:
        response = index.upsert(vectors=[{
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "product_id": product_id,
                "category": category
            }
        }])

        # Print or log success info
        upserted = response.get("upserted_count", 0)
        print(f"✅ Upsert complete: {upserted} vector(s) added or updated (ID: {vector_id})")

        return {
            "success": True,
            "upserted_count": upserted,
            "values": embedding,
            "vector_id": vector_id
        }

    except Exception as e:
        print(f"❌ Error during upsert: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Query example
def query_similar(embedding: list, category: str, top_k: int = 5):
    try:
        result = index.query(
            vector=embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            filter={"category": category}
        )

        matches = result.get('matches', [])
        if not matches:
            print("No similar items found.")
            return []

        return [
        {"id": match["id"], "score": match.get("score"), "metadata": match.get("metadata")}
        for match in matches
        ]


    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []


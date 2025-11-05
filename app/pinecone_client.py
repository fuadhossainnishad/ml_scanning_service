import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "learning"
EMBEDDING_DIM = 1024
CLOUD = "aws"
REGION = "us-east-1"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check existing indexes
# existing_indexes = [idx["name"] if isinstance(idx, dict) else idx for idx in pc.list_indexes()]
# print("Existing indexes:", existing_indexes)
existing_indexes_raw = pc.list_indexes()
print("Raw indexes:", existing_indexes_raw)


# Normalize to strings
existing_indexes = []
for idx in existing_indexes_raw:
    if isinstance(idx, dict):
        existing_indexes.append(idx.get("name"))
    else:
        existing_indexes.append(idx)

print("Normalized index names:", existing_indexes)
print("Raw list from Pinecone:", pc.list_indexes())
print("Processed index names:", existing_indexes)
print("Check if index exists:", INDEX_NAME in existing_indexes)


# Create index if it doesn't exist
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

# Connect to the index
index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index '{INDEX_NAME}'")

# Upsert example
def upsert_vector(product_id: str, embedding: list, category: str):
    index.upsert(vectors=[{
        "id": product_id,
        "values": embedding,
        "metadata": {"category": category}
    }])

# Query example
def query_similar(embedding: list, category: str, top_k: int = 5):
    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_values=False,
        filter={"category": category}
    )
    return [match['id'] for match in result['matches']]

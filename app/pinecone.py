from pinecone import Pinecone, ServerlessSpec
from app.config import config

# Initialize Pinecone
pc = Pinecone(api_key=config["pinecone_api_key"])

# Create or connect index
INDEX_NAME = "clothes"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=config["embedding_dim"],
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=config["pinecone_region"])
    )

index = pc.Index(INDEX_NAME)

# Upsert vector with metadata
def upsert_vector(product_id, embedding, category):
    index.upsert(vectors=[{
        "id": product_id,
        "values": embedding,
        "metadata": {"category": category}
    }])

# Query similar vectors by embedding + category
def query_similar(embedding, category, top_k=5):
    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_values=False,
        filter={"category": category}
    )
    return [match['id'] for match in result['matches']]

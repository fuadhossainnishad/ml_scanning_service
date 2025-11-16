from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from app.pinecone_client import upsert_vector
from app.embeddings import get_image_embedding
import uuid
# from typing import List
router = APIRouter()


@router.post('/insert')
async def upsert_image(
    file: UploadFile = File(...),
    product_id: str = Query(..., description="Unique product ID"),
    category: str = Query(..., description="Category to filter"),
) -> dict:
    try:
        image_bytes = await file.read()
        embedding = get_image_embedding(image_bytes)
        await file.close()

        # Generate a unique ID for this image
        vector_id = str(uuid.uuid4())

        # Upsert vector to Pinecone
        upsert_vector(vector_id, product_id, embedding, category)

        # Now query for similar items
        # matches: List[str] = query_similar(embedding, category, top_k=top_k)

        print("status:", "success")
        print("upserted_id:", product_id)
        # print("results:", matches)

        return {
            "status": "success",
            "product_id": product_id,
            "vector_id": vector_id,
            "category": category,
            # "results": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

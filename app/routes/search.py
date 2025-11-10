from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from app.embeddings import get_image_embedding
from app.pinecone_client import query_similar
from typing import List
# import uuid

router = APIRouter()


@router.post("/scan")
async def search_by_image(
    file: UploadFile = File(...),
    # product_id: str = Query(..., description="Unique product ID"),
    category: str = Query("clothes", description="Category to filter"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return")
) -> dict:
    try:
        image_bytes = await file.read()
        embedding = get_image_embedding(image_bytes)
        await file.close()

        # Generate a unique ID for this image
        # vector_id = str(uuid.uuid4())

        # Upsert vector to Pinecone
        # upsert_vector(vector_id, product_id, embedding, category)

        # Now query for similar items
        matches: List[str] = query_similar(embedding, category, top_k=top_k)

        print("status:", "success")
        # print("upserted_id:", product_id)
        print("results:", matches)

        return {
            "status": "success",
            # "upserted_id": product_id,
            "results": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# app/routes/search.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from app.embeddings import get_image_embedding
from app.pinecone_client import query_similar
from typing import List

router = APIRouter()

@router.post("/scan")
async def search_by_image(
    file: UploadFile = File(...), 
    category: str = Query("clothes", description="Category to filter"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return")
) -> dict:
    """
    Upload an image and find similar products in the specified category.
    """
    try:
        image_bytes = await file.read()
        embedding = get_image_embedding(image_bytes)
        matches: List[str] = query_similar(embedding, category, top_k=top_k)
        await file.close()

        return {
            "status": "success",
            "results": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import search
from app.routes import upsert
# from app.pinecone_client import pc, INDEX_NAME, init_index

# Use lifespan to handle startup/shutdown events (recommended)
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Initialize Pinecone index
        init_index()
    except Exception as e:
        print("Pinecone setup error:", e)
    yield
    # Optional shutdown code can go here
    print("Shutting down FastAPI...")

# Single FastAPI instance
app = FastAPI(
    title="Clothing Similarity API",
    description="Scan a clothing image and find matching products.",
    version="1.0.0",
    lifespan=lifespan
)


# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

# Include your search router
app.include_router(search.router, prefix="/api/v1")
app.include_router(upsert.router, prefix="/api/v1")

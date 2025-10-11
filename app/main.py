from fastapi import FastAPI

app=FastAPI(
    title="My FastAPI Application",
    description="This is a sample FastAPI application.",
    version="1.0.0",
)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/api/v1")
def read_root():
    return {"message": "Welcome to My FastAPI Application!"}
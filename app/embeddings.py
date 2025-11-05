# app/embeddings.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import numpy as np

# Load CLIP once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_bytes: bytes) -> list:
    """Takes raw image bytes and returns normalized CLIP embedding as list"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

    return emb.cpu().numpy().tolist()[0]

from transformers import CLIPModel, CLIPProcessor
import os


def load_clip_model(model_path: str, device: str):
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model not found at '{model_path}'. "
            f"Run 'python download_model.py' first."
        )

    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.eval()

    print(f"✅ Fashion-CLIP loaded from {model_path} on {device}")
    return model, processor





# from transformers import CLIPModel, CLIPProcessor
# import torch


# def load_clip_model(model_path: str, device: str):
#     model = CLIPModel.from_pretrained(model_path).to(device)
#     processor = CLIPProcessor.from_pretrained(model_path)
#     model.eval()
#     print(f"✅ Model loaded from {model_path} on {device}")
#     return model, processor


# from transformers import CLIPProcessor, CLIPModel
# import os


# def load_clip_model(LOCAL_MODEL_PATH: str, device):
#     """
#     Load CLIP model and processor.
#     If the local path exists, load from it.
#     Otherwise, download from HuggingFace and cache automatically.
#     """
#     if os.path.exists(LOCAL_MODEL_PATH):
#         model = CLIPModel.from_pretrained(LOCAL_MODEL_PATH).to(device)
#         processor = CLIPProcessor.from_pretrained(LOCAL_MODEL_PATH)
#         print(f"✅ Loaded CLIP model from local path: {LOCAL_MODEL_PATH}")
#     else:
#         print("⚠️ Local model not found. Downloading from HuggingFace...")
#         model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#         processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         print("✅ Downloaded CLIP model from HuggingFace")
#     return model, processor

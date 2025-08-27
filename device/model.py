import os
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
from config import MODEL_PATH

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 模型文件不存在: {MODEL_PATH}")

print("✅ 正在加载模型:", MODEL_PATH)
session = ort.InferenceSession(MODEL_PATH)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).numpy()
    emb = session.run(None, {"input": x})[0]
    # 转为 1D 向量并归一化
    emb = emb.flatten()
    return emb / np.linalg.norm(emb)

import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

session = ort.InferenceSession("mobilenet_feature_extractor.onnx")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).numpy()
    emb = session.run(None, {"input": x})[0]
    return emb / np.linalg.norm(emb)

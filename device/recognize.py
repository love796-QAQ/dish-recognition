import numpy as np
from model import get_embedding
from templates import template_db
from config import SIMILARITY_THRESHOLD

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def recognize(img_path, top_k=3):
    emb = get_embedding(img_path).flatten()

    results = []
    for dish, embs in template_db.items():
        if not embs:
            continue
        scores = [cosine_similarity(emb, e) for e in embs]
        score = max(scores)
        results.append((dish, score))

    # 排序并过滤低相似度
    results.sort(key=lambda x: x[1], reverse=True)
    results = [(dish, score) for dish, score in results if score >= SIMILARITY_THRESHOLD]

    return results[:top_k]

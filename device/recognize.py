import numpy as np
from model import get_embedding
from templates import template_db

def recognize(img_path):
    emb = get_embedding(img_path)
    best_dish, best_score = None, -1
    for dish, embs in template_db.items():
        scores = [np.dot(emb, e.T).item() for e in embs]
        score = max(scores)
        if score > best_score:
            best_dish, best_score = dish, score
    return best_dish, best_score

import numpy as np
import pickle
from model import get_embedding

template_db = {}

def add_template(dish_name, img_path):
    emb = get_embedding(img_path)
    template_db.setdefault(dish_name, []).append(emb)
    print(f"模板已添加: {dish_name}, 数量={len(template_db[dish_name])}")

def save_templates(path="templates.pkl"):
    with open(path, "wb") as f:
        pickle.dump(template_db, f)

def load_templates(path="templates.pkl"):
    global template_db
    with open(path, "rb") as f:
        template_db = pickle.load(f)

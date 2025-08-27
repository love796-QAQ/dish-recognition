import os
from model import get_embedding
from config import TEMPLATE_DIR

template_db = {}

def load_templates():
    base_dir = TEMPLATE_DIR
    if not os.path.exists(base_dir):
        print(f"❌ 模板目录不存在: {base_dir}")
        return {}

    for dish_folder in os.listdir(base_dir):
        dish_path = os.path.join(base_dir, dish_folder)
        if not os.path.isdir(dish_path):
            continue
        dish_name = dish_folder
        template_db[dish_name] = []
        for f in os.listdir(dish_path):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(dish_path, f)
                emb = get_embedding(img_path)
                template_db[dish_name].append(emb)
        print(f"已录入模板: {dish_name}, 图片数量={len(template_db[dish_name])}")

    return template_db

import sys
sys.dont_write_bytecode = True  # 禁止生成 __pycache__

import os
from templates import load_templates, template_db
from recognize import recognize

def main():
    print("=== 🍲 菜品识别系统 ===")

    load_templates()
    if not template_db:
        print("⚠️ 没有加载到任何模板，请确认 TEMPLATE_DIR 下有子文件夹和图片")
        return

    while True:
        print("\n操作菜单：")
        print("1. 输入图片绝对路径进行识别")
        print("2. 查看已加载的模板")
        print("3. 退出")

        choice = input("请输入操作编号: ").strip()

        if choice == "1":
            img_path = input("请输入图片绝对路径: ").strip().strip('"').strip("'")
            if not os.path.exists(img_path):
                print(f"❌ 找不到文件: {img_path}")
                continue

            results = recognize(img_path, top_k=3)

            if not results:
                print("⚠️ 未识别到已知菜品（相似度过低）")
            else:
                print("✅ 识别结果（Top-3 相似度）：")
                for rank, (dish, score) in enumerate(results, start=1):
                    print(f" {rank}. {dish} | 相似度={score:.3f}")

        elif choice == "2":
            if not template_db:
                print("📂 暂无模板")
            else:
                print("📊 已加载的模板：")
                for dish, embs in template_db.items():
                    print(f"- {dish} (图片数量: {len(embs)})")

        elif choice == "3":
            print("👋 程序已退出")
            break

if __name__ == "__main__":
    main()

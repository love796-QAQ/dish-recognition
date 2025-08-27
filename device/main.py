from templates import add_template, save_templates, load_templates
from scale import add_dish

# 示例流程
if __name__ == "__main__":
    # 加载历史模板（可选）
    try:
        load_templates()
    except:
        pass

    # 新增模板
    add_template("宫保鸡丁", "gongbao1.jpg")
    add_template("鱼香肉丝", "yuxiang1.jpg")
    save_templates()

    # 识别 & 称重
    print(add_dish("gongbao_test.jpg", 200))
    print(add_dish("yuxiang_test.jpg", 350))

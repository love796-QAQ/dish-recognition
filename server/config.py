# server/config.py

# 训练集根目录 (里面必须有 train/ 和 val/ 文件夹)
DATASET_DIR = r"C:\Users\wangshuo523\Desktop\test\datasets"

# 模型保存目录
MODEL_DIR = r"C:\Users\wangshuo523\Desktop\test"

# 默认训练轮数
NUM_EPOCHS = 100

# 早停法参数: 容忍验证集 Loss 连续多少次未下降
PATIENCE = 5
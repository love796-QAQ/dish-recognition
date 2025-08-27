import sys
sys.dont_write_bytecode = True  # 禁止生成 __pycache__

import os
import torch
import torch.nn as nn
from train import train_model
from torchvision import models
from config import MODEL_DIR

def export_to_onnx(model, num_classes):
    os.makedirs(MODEL_DIR, exist_ok=True)
    onnx_path = os.path.join(MODEL_DIR, "mobilenet_feature_extractor.onnx")

    base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    base_model.classifier[1] = torch.nn.Linear(base_model.last_channel, num_classes)
    base_model.load_state_dict(model.state_dict())
    base_model.eval()

    feature_extractor = torch.nn.Sequential(
        base_model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(1)
    )

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(feature_extractor, dummy_input, onnx_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

    print(f"✅ 模型已转换为 ONNX (1280维特征向量): {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    model, num_classes = train_model()
    onnx_file = export_to_onnx(model, num_classes=num_classes)

    print("✅ 训练与导出流程完成")
    print(f"📂 你可以在 {MODEL_DIR} 中找到模型文件：")
    print(f" - mobilenet_classifier.pth")
    print(f" - mobilenet_feature_extractor.onnx")
    print("\n👉 请在 device/config.py 中手动配置 MODEL_PATH 使用该模型。")

import torch, torch.nn as nn
from torchvision import models

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)  # 类别数随便写，加载时会替换
model.load_state_dict(torch.load("mobilenet_classifier.pth", map_location="cpu"))

feature_extractor = nn.Sequential(
    model.features,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
)

torch.save(feature_extractor.state_dict(), "mobilenet_feature_extractor.pth")

dummy = torch.randn(1,3,224,224)
torch.onnx.export(feature_extractor, dummy, "mobilenet_feature_extractor.onnx",
                  input_names=["input"], output_names=["embedding"], opset_version=11)
print("特征提取模型已导出")

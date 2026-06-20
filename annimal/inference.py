import argparse
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ====== 配置 ======
DATA_PATH = "./raw-img"
MODEL_PATH = "models/resnet18_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME_MAP = {
    'cane': '🐕 狗',
    'cavallo': '🐎 马',
    'elefante': '🐘 大象',
    'farfalla': '🦋 蝴蝶',
    'gallina': '🐔 鸡',
    'gatto': '🐈 猫',
    'mucca': '🐄 牛',
    'pecora': '🐑 羊',
    'ragno': '🕷️ 蜘蛛',
    'scoiattolo': '🐿️ 松鼠'
}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_class_names() -> Tuple[List[str], List[str]]:
    """读取 raw-img 目录中的类别，并返回原始类别名和中文显示名。"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到数据目录 {DATA_PATH}")
    class_names_raw = sorted([
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    ])
    class_names_display = [NAME_MAP.get(name, name) for name in class_names_raw]
    return class_names_raw, class_names_display


def load_model(model_path: str = MODEL_PATH) -> torch.nn.Module:
    """加载训练好的 ResNet18 模型。"""
    _, class_names_display = get_class_names()
    num_classes = len(class_names_display)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件 {model_path}，请先运行 train.py 训练模型。")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(img: Image.Image, model: torch.nn.Module, topk: int = 3) -> List[Tuple[str, float]]:
    """对 PIL.Image 进行推理，返回 top-k 预测结果。"""
    _, class_names_display = get_class_names()
    img_tensor = TRANSFORM(img.convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

    topk = min(topk, len(class_names_display))
    ranked = sorted(
        [(class_names_display[i], float(probs[i])) for i in range(len(class_names_display))],
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:topk]


def predict_image_from_path(image_path: str, model: torch.nn.Module, topk: int = 3) -> List[Tuple[str, float]]:
    """对本地图片路径进行识别。"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片文件 {image_path}")
    img = Image.open(image_path)
    return predict_image(img, model, topk)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地图片动物识别")
    parser.add_argument("image", help="本地图片路径，例如 ./test.jpg")
    parser.add_argument("--topk", type=int, default=3, help="显示前 K 个预测结果")
    args = parser.parse_args()

    model = load_model()
    results = predict_image_from_path(args.image, model, topk=args.topk)
    print(f"输入图片: {args.image}")
    print(f"Top-{args.topk} 预测结果:")
    for label, prob in results:
        print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()

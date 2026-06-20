# evaluate.py 
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from tqdm import tqdm  # 导入进度条

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

print("🚀 开始评估模型...")

DATA_PATH = "./raw-img"
MODEL_PATH = "models/resnet18_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 检查文件
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到模型文件 {MODEL_PATH}")

# 读取类别
class_names_raw = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
NUM_CLASSES = len(class_names_raw)

# 意大利语 → 中文
NAME_MAP = {
    'cane': '狗', 'cavallo': '马', 'elefante': '大象',
    'farfalla': '蝴蝶', 'gallina': '鸡', 'gatto': '猫',
    'mucca': '牛', 'pecora': '羊', 'ragno': '蜘蛛', 'scoiattolo': '松鼠'
}
class_names_cn = [NAME_MAP.get(name, name) for name in class_names_raw]

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

print("📂 正在加载数据集（共约 28,000 张图片，请耐心等待）...")
dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

# 加载模型
print("🧠 加载模型...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("🔍 开始推理（进度条会显示当前进度，约需 3~5 分钟）...")
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(loader, desc="推理进度", unit="批"):
        images = images.to(DEVICE)
        preds = model(images).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("✅ 推理完成，正在生成混淆矩阵...")

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names_cn, yticklabels=class_names_cn)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

print("\n📊 分类报告:")
print(classification_report(all_labels, all_preds, target_names=class_names_cn))

acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n🎯 最终测试集准确率: {acc:.2%}")
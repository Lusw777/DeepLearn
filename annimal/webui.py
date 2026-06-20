import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ==================== 配置 ====================
DATA_PATH = "./raw-img"  
MODEL_PATH = "models/resnet18_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 检查数据集
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ 找不到 {DATA_PATH}，请检查路径！")

# 自动读取文件夹名作为类别
class_names_raw = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
NUM_CLASSES = len(class_names_raw)

# 意大利语 → 中文 映射表（用于网页显示）
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
class_names_display = [NAME_MAP.get(name, name) for name in class_names_raw]
print(f"📂 识别类别: {class_names_display}")

# ==================== 加载模型 ====================
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 找不到模型 {MODEL_PATH}，请先运行 train.py 训练！")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()
print("✅ 模型加载成功！")

# ==================== 预处理 ====================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 预测函数 ====================
def predict_image(img):
    if img is None:
        return {name: 0.0 for name in class_names_display}
    try:
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        # 用中文名作为键返回
        return {class_names_display[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    except Exception as e:
        return {"错误": f"{str(e)}"}

# ==================== 界面 ====================
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="📤 点击上传动物图片"),
    outputs=gr.Label(num_top_classes=3, label="🎯 识别结果 (Top-3)"),
    title="🐾 动物识别系统 - 深度学习课程设计",
    description="""
    <div style="text-align: center;">
        <p>基于 <strong>ResNet18</strong> 卷积神经网络</p>
        <p>可识别：狗、猫、马、蜘蛛、蝴蝶、鸡、羊、牛、松鼠、大象</p>
    </div>
    """
)

if __name__ == "__main__":
    print("🚀 正在启动网页，浏览器即将弹出...")
    interface.launch(share=True, inbrowser=True)
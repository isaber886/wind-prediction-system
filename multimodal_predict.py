import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # 【新增】专门用于调用 Softmax 函数
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# ==========================================
# 第一部分：利用训练好的 ResNet-18 提取图像软标签 (Softmax 概率)
# ==========================================
print("阶段 1：正在加载训练好的 ResNet 图像特征提取器...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 重新构建与训练时一模一样的模型结构
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 之前设定了 3 个分类

# 加载你训练好的权重文件
model.load_state_dict(torch.load('resnet18_wind_classifier.pth', map_location=device))
model = model.to(device)
model.eval()  # 【关键】设置为评估模式，防止 Dropout 影响预测并停止梯度更新

# 图像预处理 (必须和训练阶段完全一致)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 读取多模态标签表
df = pd.read_csv('multimodal_labels.csv')
image_dir = 'dataset_images'

# 准备 3 个空列表，用于存放 3 种气象模态的置信度概率
cnn_prob_0 = []
cnn_prob_1 = []
cnn_prob_2 = []

print("阶段 2：正在通过 CNN 提取软标签概率特征 (解决模型坍塌)...")
# 禁用梯度计算，极大地加速推理过程并节省显存
with torch.no_grad():
    for img_name in tqdm(df['image_id']):
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

        # 前向传播获取原始 Logits 输出
        outputs = model(image)

        # 【核心修改点：Softmax 软标签提取】
        # 将无界的 Logits 转化为 0~1 之间的置信度概率分布
        probabilities = F.softmax(outputs, dim=1)[0]

        # 将三个类别的概率分别保存
        cnn_prob_0.append(probabilities[0].item())
        cnn_prob_1.append(probabilities[1].item())
        cnn_prob_2.append(probabilities[2].item())

# 将提取到的概率作为连续的数值特征，正式加入数据表
df['cnn_prob_low'] = cnn_prob_0  # 低效风况的置信度
df['cnn_prob_normal'] = cnn_prob_1  # 正常风况的置信度
df['cnn_prob_high'] = cnn_prob_2  # 满发风况的置信度

# ==========================================
# 第二部分：多模态特征融合与随机森林功率预测
# ==========================================
print("\n阶段 3：开始多模态特征融合与随机森林预测...")

# 1. 准备特征矩阵 (X) 和目标变量 (y)
# 【核心组合】传统数值气象特征 + CNN提取的图像气象模态概率
feature_cols = [
    'feature_wind_speed',
    'feature_wind_dir',
    'feature_wind_std',
    'cnn_prob_low',
    'cnn_prob_normal',
    'cnn_prob_high'
]

X = df[feature_cols].values
y = df['target_power'].values

# 2. 划分训练集和测试集 (80% 训练，20% 验证)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# 4. 在测试集上进行盲测预测
y_pred = xgb_model.predict(X_test)

# 5. 模型终极评估
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 40)
print("🎯 最终级联预测系统评估结果 (Random Forest):")
print(f"平均绝对误差 (MAE):  {mae:.2f} kW")
print(f"均方根误差 (RMSE): {rmse:.2f} kW")
print(f"决定系数 (R² Score): {r2:.4f}")
print("=" * 40)

# 5. 查看特征重要性并打印 (xgb_model)
print("\n🔍 融合特征重要性分析 (Feature Importances):")
importances = xgb_model.feature_importances_

# 将特征名和重要性组合并按权重降序排列
feature_importance_dict = dict(zip(feature_cols, importances))
sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

for feature, imp in sorted_features:
    print(f"  - {feature:<20}: {imp:.4f} ({imp * 100:.2f}%)")

joblib.dump(xgb_model, 'wind_xgb_model.pkl')
print("机器学习模型已保存为 wind_xgb_model.pkl")

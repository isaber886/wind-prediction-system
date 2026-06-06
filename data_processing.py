import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# 1. 读取数据 (根据你的实际路径修改)
df = pd.read_csv('data/T1.csv')
df.columns = ['Time', 'ActivePower', 'WindSpeed', 'TheoreticalPower', 'WindDirection']
df['Time'] = pd.to_datetime(df['Time'], format='%d %m %Y %H:%M')
df.set_index('Time', inplace=True)

print(f"初始数据量: {len(df)}")

# ----------------- 步骤A：进阶清洗 (清洗异常运行状态) -----------------
# 1. 消除负功率
df['ActivePower'] = df['ActivePower'].apply(lambda x: max(0, x))

# 2. 【核心】风功率曲线过滤：剔除实际功率与理论功率偏差超过 20% 且风速 > 3m/s 的限电异常点
# 理论功率为0时避免除以0报错，加上一个极小值 1e-5
error_ratio = abs(df['ActivePower'] - df['TheoreticalPower']) / (df['TheoreticalPower'] + 1e-5)
limit_power_mask = (df['WindSpeed'] > 3.0) & (error_ratio > 0.20)
df_cleaned = df[~limit_power_mask].copy()

# 3. 线性插值处理缺失值
df_cleaned = df_cleaned.interpolate(method='linear')

print(f"进阶清洗后数据量: {len(df_cleaned)}")

# ----------------- 步骤B：特征归一化 (为画图和神经网络做准备) -----------------
# 实例化归一化器，将数值映射到 0 到 1 之间
scaler = MinMaxScaler()

# 选取需要归一化的特征列
features_to_scale = ['WindSpeed', 'WindDirection', 'ActivePower']
df_cleaned[features_to_scale] = scaler.fit_transform(df_cleaned[features_to_scale])

# ----------------- 步骤C：时序特征衍生 (为最后的数值预测阶段做准备) -----------------
# 计算过去 6 个时间步（即过去 1 小时）的风速方差，代表“阵风波动强度”
df_cleaned['WindSpeed_Std_1h'] = df_cleaned['WindSpeed'].rolling(window=6).std()

# 剔除因为 rolling 产生的开头 NaN 值
df_cleaned.dropna(inplace=True)

print("数据处理与归一化完成！现在的 df_cleaned 可以极其完美地用于生成 PNG 图像了。")

# 1. 创建存放数据集的目录
image_dir = 'dataset_images'
os.makedirs(image_dir, exist_ok=True)

# 2. 参数设置
window_size = 144  # 过去24小时的数据点 (144 * 10分钟)
label_records = []  # 用于存放每一张图对应的表格数据

print("开始生成特征图像与标签映射表...")

# 3. 滑动窗口生成图像 (这里以生成前 500 个样本为例，跑通后可换成 len(df_cleaned) - window_size)
total_samples = 500

for i in tqdm(range(total_samples)):
    # 截取滑动窗口内的数据
    window_data = df_cleaned.iloc[i: i + window_size]

    # 【预测目标】：我们用这 24 小时的数据，来预测紧接着的下一个时间点（第145个点）的实际功率
    target_power = df_cleaned['ActivePower'].iloc[i + window_size]

    # 【保留数值特征】：提取窗口最后一个时间点的数值，后续和图像特征一起喂给模型
    current_wind_speed = window_data['WindSpeed'].iloc[-1]
    current_wind_dir = window_data['WindDirection'].iloc[-1]
    current_wind_std = window_data['WindSpeed_Std_1h'].iloc[-1]

    # ---------------- 绘图环节 ----------------
    # 创建 2.24 x 2.24 英寸，DPI=100 的画布，生成刚好 224x224 像素的图片 (ResNet 的标准输入)
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)

    # 绘制归一化后的风速(蓝线)和风向(红线半透明)
    ax.plot(window_data['WindSpeed'].values, color='blue', linewidth=2)
    ax.plot(window_data['WindDirection'].values, color='red', alpha=0.5)

    ax.axis('off')

    # 保存图片
    img_filename = f"seq_{i}.png"
    img_path = os.path.join(image_dir, img_filename)
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)  # 必须关闭画布，释放内存

    # ---------------- 记录映射关系 ----------------
    label_records.append({
        'image_id': img_filename,
        'feature_wind_speed': current_wind_speed,
        'feature_wind_dir': current_wind_dir,
        'feature_wind_std': current_wind_std,
        'target_power': target_power
    })

# 4. 保存为多模态映射表
labels_df = pd.DataFrame(label_records)
labels_df.to_csv('multimodal_labels.csv', index=False)

print("\n图像构建完成！")
print(f"图片保存在 {image_dir} 文件夹下。")
print(f"特征与标签文件 multimodal_labels.csv 已生成。")
df = pd.read_csv('multimodal_labels.csv')
print(df['target_power'].describe()) # 看一下 25%, 50%, 75% 的分位数


import matplotlib.pyplot as plt
import numpy as np

# 定义时序步长
time_steps = np.arange(0.2, 2.2, 0.2)

# # 假设的混淆矩阵值随时间变化
# TP = np.linspace(90, 100, 10)  # True Positives
# TN = np.linspace(50, 60, 10)   # True Negatives
# FP = np.linspace(10, 5, 10)    # False Positives
# FN = np.linspace(20, 15, 10)   # False Negatives

# 假设的混淆矩阵值随时间变化
TP = np.array([105,106,103,112,117,110,105,99,92,83])  # True Positives  表示模型正确预测攻击序列的次数
FP =  np.array([120-v for v in TP])    # False Positives 模型错误预测为攻击序列的次数（实际为正常序列）
TN = np.array([102,105,113,110,114,102,96,86,88,72])  # True Negatives 模型正确预测正常序列的次数
FN = np.array([120-v for v in TN])   # False Negatives 模型错误预测为正常序列的次数（实际为攻击序列）

print("TP",TP)
print("FP",FP)
print("TN",TN)
print("FN",FN)

# 计算指标
precision = TP / (TP + FP)*100
recall = TP / (TP + FN)*100
fnr = FN / (TP + FN)*100
fpr = FP / (FP + TN)*100

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(time_steps, precision, marker='o', linestyle='-', color='b', label='Precision')
plt.plot(time_steps, recall, marker='s', linestyle='-', color='r', label='Recall')
plt.plot(time_steps, fnr, marker='^', linestyle='-', color='g', label='FNR')
plt.plot(time_steps, fpr, marker='x', linestyle='-', color='y', label='FPR')

# plt.title('Performance Metrics Over Time Based on Confusion Matrix')
plt.xlabel('Time Steps')
plt.ylabel('Percentage')
plt.legend()
plt.grid(True)
plt.show()
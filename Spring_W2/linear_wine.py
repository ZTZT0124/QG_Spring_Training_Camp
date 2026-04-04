import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import *

data = pd.read_csv('winequality-red.csv', delimiter=';')
X = data.drop("quality", axis=1).values  # 分离quality的值和其他特征值
y = data["quality"].values
print(y)
X_features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
# 分割数据
X_train,y_train,X_val,y_val,X_test,y_test = train_val_test_split(X, y)

# 特征标准化
X_train_poly = np.c_[X_train, X_train**2, X_train**3]  # 扩展
X_train, mu_train, sigma_train = zscore_normalize_features(X_train_poly)  # 标准化

# 验证集/测试集必须和训练集保持一致！
X_val_poly = np.c_[X_val, X_val**2, X_val**3]
X_val = (X_val_poly - mu_train) / sigma_train  # 用训练集的mu/sigma

X_test_poly = np.c_[X_test, X_test**2, X_test**3]
X_test = (X_test_poly - mu_train) / sigma_train
m, n = X_train.shape
w_init = np.zeros(n)
b_init = 0
alpha =0.01
num_iter = 2000
lambda_ = 0.1
# 开始梯度下降
w_final,b_final, J_train_history, J_val_history = gradient_descent(X_train, y_train,X_val,y_val, w_init, b_init, alpha, num_iter, lambda_)

pred_y_train = np.dot(X_train, w_final) + b_final
pred_y_test = np.dot(X_test, w_final) + b_final

def compute_mse(y_true, y_pred):
    m = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / m


def compute_mae(y_true, y_pred):
    m = len(y_true)
    return np.sum(np.abs(y_true - y_pred)) / m


def compute_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_total)

# 计算指标
train_metrics = {
    "均方误差": compute_mse(y_train, pred_y_train),
    "平均绝对误差": compute_mae(y_train, pred_y_train),
    "决定系数": compute_r2(y_train, pred_y_train)
}
test_metrics = {
    "均方误差": compute_mse(y_test, pred_y_test),
    "平均绝对误差": compute_mae(y_test, pred_y_test),
    "决定系数": compute_r2(y_test, pred_y_test)
}

# 打印评估结果
print("训练集评估:")
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")
print("测试集评估:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")


fig, ax = plt.subplots(1,1,figsize=(8,8))

ax.plot(np.arange(0,len(J_train_history)*20,20),J_train_history,color='blue',label = 'Train Cost')
ax.plot(np.arange(0,len(J_val_history)*20,20),J_val_history,color='red',label = 'Val Cost')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_ylim([0,0.75])
ax.legend()



fig2, ax2 = plt.subplots(3,4,figsize=(15,12))
ax2 = ax2.flatten()
for i in range(int(n/3)):
    ax2[i].scatter(X_train[:,i],pred_y_train,color='blue',label='Predicted')
    ax2[i].scatter(X_train[:,i],y_train,color='red',label='Actual')
ax2[0].legend()

# 补充：全局视角看预测效果
fig4, ax4 = plt.subplots(1,1,figsize=(8,6))
ax4.scatter(y_test, pred_y_test, alpha=0.5)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 完美预测线
ax4.set_xlabel('Real Quality')
ax4.set_ylabel('Perdict Quality')
ax4.set_title('Test')
plt.show()


plt.show()



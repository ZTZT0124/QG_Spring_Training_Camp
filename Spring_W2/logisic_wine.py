import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from logistic_regression import *

data = pd.read_csv('winequality-red.csv', delimiter=';')
X = data.drop("quality", axis=1).values  # 分离quality的值和其他特征值
y_in = data["quality"].values
X_features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]

m, n = X.shape  # m个样本，n个特征
y = copy.deepcopy(y_in)
for i in range(m):
    if y[i] > 6:
        y[i] = 1
    else:
        y[i] = 0

# 分离数据
X_train,y_train,X_val,y_val,X_test,y_test = train_val_test_split(X,y)
# 标准化数据
X_train,mu,sigma = zscore_normalize_features(X_train)
X_val = (X_val - mu) / sigma
X_test = (X_test - mu) / sigma
print("数据标准化完成！")

w_init = np.zeros(n)  # 对每个特征有一个w
b_init = 0.0
alpha =0.1
num_iters =5000
lambda_ = 0.1
w_final, b_final, J_train_history, J_val_history = gradient_logistic(X_train, y_train,X_val, y_val,  w_init, b_init, alpha, num_iters,lambda_)
print("训练完成！开始评估！")

# 判读是否过拟合/欠拟合，即看J_train & J_val的大小
final_J_train = J_train_history[len(J_train_history)-1]
final_J_val = J_val_history[len(J_val_history)-1]
cost_gap = final_J_val - final_J_train

print("模型评估结果：")
print(f"最终的训练集代价大小为：{final_J_train:.4f}")
print(f"最终的验证集代价大小为：{final_J_val:.4f}")
print(f"两者的差值为：{cost_gap:.4f}")

# 开始对模型进行测试，用测试集的数据，计算模型的测试集代价
test_f_wb = sigmoid(np.dot(X_test,w_final) + b_final)  # 得到的是跟测试集一样维度的矩阵
test_pred_label = np.where(test_f_wb >= 0.5, 1, 0)

accuracy,good2bad, bad2good = evaluate_model(y_test,test_pred_label)

accuracy*=100
good2bad*=100
bad2good *=100


print(f'模型预测的正确率为{accuracy:.4f}%')
print(f'把好酒判断为坏酒的概率为：{good2bad:.4f}%')
print(f'把坏酒判断为好酒的概率为：{bad2good:.4f}%')

# 可视化
fig1, ax1 = plt.subplots(1,1,figsize=(10,5))
iterations = np.arange(0,len(J_train_history) * 20, 20)
ax1.plot(iterations, J_train_history,color='blue',label='training cost')
ax1.plot(iterations, J_val_history,color='red',label='val cost')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Cost')
ax1.legend(loc='best')
ax1.set_title("is the model accuracy")


fig3, ax3 = plt.subplots(3,4,figsize=(16,12))
ax3 = ax3.flatten()
for i in range(n):
    ax3[i].scatter(X_train[:,i],y_train,color = 'red', alpha = 0.5,label='ture label')
    ax3[i].scatter(X_train[:,i],sigmoid(np.dot(X_train, w_final)+ b_final),color='blue',label='predict label')
    ax3[i].set_xlabel(X_features[i])
    ax3[i].set_ylim([-0.1,1.1])

ax3[0].set_ylabel('predict label')
ax3[0].legend()
fig3.suptitle('the real labels and predict labels')
fig3.tight_layout()



plt.show()

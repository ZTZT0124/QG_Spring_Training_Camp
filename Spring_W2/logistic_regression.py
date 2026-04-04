import numpy as np
import matplotlib.pyplot as plt
import copy


def sigmoid(z):
    return 1/(1+np.exp(-z))

# 求代价函数的值
def compute_cost_logistic(X, y, w, b,lambda_ = 0):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost /= m

    n = w.shape[0]
    reg_cost = 0.0
    for j in range(n):
        reg_cost += w[j]**2
    reg_cost = reg_cost * (lambda_/(2 * m))

    cost += reg_cost
    return cost

# 求偏导
def compute_gradient_logistic(X, y, w, b, lambda_ = 0):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j]+err * X[i, j]
        dj_db += err

    dj_dw /= m
    dj_db /=m

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]

    return dj_dw, dj_db

# 梯度下降
def gradient_logistic(X_train, y_train, X_val, y_val,  w_in, b_in, alpha, num_iters,lambda_ =  0):
    J_train_history = []
    J_val_history = []
    w = copy.deepcopy(w_in)
    b = copy.deepcopy(b_in)

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X_train, y_train, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % 20 == 0:
            J_train_history.append(compute_cost_logistic(X_train, y_train, w, b, lambda_))
            J_val_history.append(compute_cost_logistic(X_val, y_val, w, b, lambda_))

        if len(J_train_history) >= 5:
            if all(J_val_history[-1] >= J_val_history[-i-1] for i in range(5)):
                print("验证集代价函数不再下降，迭代停止")
                break

    return w, b, J_train_history, J_val_history


# 标准化数据
def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X - mu) / sigma
    return X, mu, sigma


def train_val_test_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    m = X.shape[0]
    shuffled_indices = np.random.permutation(m)  # 打乱顺序

    train_size = int(m * train_size)
    val_size = int(m * val_size)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]

    x_train = X[train_indices]
    y_train = y[train_indices]

    x_val = X[val_indices]
    y_val = y[val_indices]

    x_test = X[test_indices]
    y_test = y[test_indices]

    return x_train, y_train, x_val, y_val, x_test, y_test

def evaluate_model(y_true,y_pred):
    m = len(y_true)
    right = 0
    error = 0
    good2bad = 0
    bad2good = 0
    for i in range(m):
        if y_true[i] == y_pred[i]:
            right += 1
        else:
            error += 1

        if y_true[i] == 1 and y_pred[i] == 0:
            good2bad += 1

        elif y_true[i] == 0 and y_pred[i] == 1:
            bad2good += 1

    epsilon = 1e-10
    accuracy = right / (m + epsilon)
    good2bad /= (m + epsilon)
    bad2good /= (m + epsilon)
    return accuracy, good2bad, bad2good
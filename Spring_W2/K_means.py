import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_regression import zscore_normalize_features

columns = ['sepal_length_in_cm', 'sepal_width_in_cm', 'petal_length_in_cm', 'petal_width_in_cm', 'class']
data = pd.read_csv('iris.data', header=None, names=columns)
X = data.drop(columns=['class']).values
y_true = data['class'].values
# print(data.head())


y_true_numeric = np.zeros(len(y_true), dtype=int)
y_true_numeric[y_true == "Iris-setosa"] = 0
y_true_numeric[y_true == "Iris-versicolor"] = 1
y_true_numeric[y_true == "Iris-virginica"] = 2

X, mu, sigma = zscore_normalize_features(X)

"""
1.数据标准化
2.选择聚类中心
3.计算距离
4.将点分配给中心
5.计算代价
6.将中心坐标更新为属于该聚类中心的所有点的均值点（每个维度坐标相加除以样本量）
7.重复步骤3. 4. 5. 6.
8.结束

"""



# 选择样本作为聚类中心
def init_centroids(X, k):
    m = X.shape[0]
    # 随机生成k个不重复的索引
    random_indices = np.random.choice(m, k, replace=False)
    centroids = X[random_indices]  # 聚类中心是一个矩阵的形式，第i行表示第i个中心，
    return centroids


# 计算距离
def calculate_distances(x1,x2):
    d =np.sqrt(np.sum((x1 - x2) ** 2))
    return d


"""
将点分配给中心
1.计算每个点到每个中心的距离
2.找最小值
3.将点标记为对应中心的点
"""


def assign_clusters(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    cluster_labels = np.zeros(m, dtype=int)  # 每个样本的聚类标签（0/1/2）
    # 遍历每个点
    for i in range(m):
        # 计算点到每个中心的距离
        distances = np.zeros(k)
        for j in range(k):
            distances[j] = calculate_distances(X[i], centroids[j])
        # 把样本分到距离最小的那个类
        cluster_labels[i] = np.argmin(distances)
    return cluster_labels


"""
将中心坐标更新为属于该聚类中心的所有点的均值点（每个维度坐标相加除以样本量）
1.计算所有样本所有维度的平均值
2.将中心点坐标更新为平均值
"""


def update_centroids(X, cluster_labels, k):
    n = X.shape[1]
    new_centroids = np.zeros((k, n))

    for j in range(k):
        # 找到所有属于第j类的样本
        cluster_samples = X[cluster_labels == j]
        # 计算平均值，作为新的中心
        new_centroids[j] = np.mean(cluster_samples, axis=0)
    return new_centroids

# 计算代价
def calculate_total_distance(X, cluster_labels, centroids):

    m = X.shape[0]
    cost = 0.0
    total_distance = 0.0
    for i in range(m):
        cost += calculate_distances(X[i], centroids[cluster_labels[i]]) ** 2
        total_distance += calculate_distances(X[i], centroids[cluster_labels[i]])
    cost /= m
    return cost


def kmeans(X, k, max_iters=1000, tol=1e-7):
    # 初始化聚类中心
    centroids = init_centroids(X, k)
    J_history = []

    for i in range(max_iters):
        # 给每个点分配聚类中心
        cluster_labels = assign_clusters(X, centroids)
        # 更新聚类中心
        new_centroids = update_centroids(X, cluster_labels, k)
        # 计算当前的代价，并保存
        cost = calculate_total_distance(X, cluster_labels, new_centroids)
        J_history.append(cost)

        # 5. 判断是否收敛：中心变化小于阈值，就停止
        centroid_change = np.sum(np.abs(new_centroids - centroids))
        if centroid_change < tol:
            break

        # 6. 更新中心，进入下一轮迭代
        centroids = new_centroids
    final_cost = J_history[-1]

    return cluster_labels, centroids, J_history, final_cost


def kmeans_multiple_runs(X, k, n_runs=50, max_iters=1000, tol = 1e-7):
    """
    多次运行K-Means，返回最优结果
    """
    # 初始化最优值：先设为无穷大，后续替换
    best_cost = float(1e8)
    best_cluster_labels = None
    best_centroids = None
    best_J_history = None
    # 记录所有运行的代价，方便后续可视化
    all_runs_cost = []

    print(f"\n开始多次运行K-Means（共{n_runs}次）...")
    for run in range(n_runs):
        # 调用单次运行函数
        cluster_labels, centroids, J_history, final_cost = kmeans(X, k, max_iters, tol)
        # 记录本次代价
        all_runs_cost.append(final_cost)

        # 对比：如果本次代价更小，更新最优结果
        if final_cost < best_cost:
            best_cost = final_cost
            best_cluster_labels = cluster_labels
            best_centroids = centroids
            best_J_history = J_history

        # 打印进度（每5次更一次）

    print(f"\n多次运行完成！最优代价：{best_cost:.6f}")
    return best_cluster_labels, best_centroids, best_J_history


k = 3
max_iters = 10000
n_runs = 50
tol = 1e-7

# 运行模型
print("\n开始K-Means聚类")
cluster_labels, centroids, J_history = kmeans_multiple_runs(X, k,n_runs, max_iters, tol)

# 查看聚类结果
print(f"每个聚类的样本数量：{np.bincount(cluster_labels)}（理想情况是50/50/50）")

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
ax1.plot(range(1, len(J_history)+1), J_history, color='blue', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('cost')
ax1.set_title('K-Means cost')
ax1.grid(alpha=0.3)


plt.show()
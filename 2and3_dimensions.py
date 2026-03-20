


import numpy as np


print("二维场景：")
v = np.array([1, 1])
x_new = np.array([1, 0])
y_new = np.array([1, 1])

# 坐标系转移
e_y = y_new / np.linalg.norm(y_new)  # 化成单位方向向量
e_x = x_new / np.linalg.norm(x_new)
# 求解线性方程组 x = ax1 + bx2; y = ay1 + by2
A = np.column_stack([e_x, e_y])
v_new = np.linalg.solve(A, v)
print("新的v向量为：[{:.9f}, {:.9f}]".format(v_new[0], v_new[1]))

# 坐标系投影
d_x = np.dot(v,x_new) / np.linalg.norm(x_new) # x轴方向投影
d_y = np.dot(v,y_new) / np.linalg.norm(y_new) # y轴方向投影
print("v向量在新x轴、新y轴方向的投影[{:.9f}, {:.9f}]".format(np.abs(d_x), np.abs(d_y)))

# 坐标系夹角 cos<v, x_new> = (点积的绝对值) / 模长的乘积
cos_x = np.dot(v, x_new) / (np.linalg.norm(x_new) * np.linalg.norm(v))
angle_x = np.arccos(cos_x)
cos_y = np.dot(v, y_new) / (np.linalg.norm(y_new) * np.linalg.norm(v))
angle_y = np.arccos(cos_y)
print("v向量与新x轴、新y轴的夹角大小：[{:.9f}, {:.9f}]".format(angle_x, angle_y))

# 坐标系面积缩放倍数
matrix = np.column_stack([x_new, y_new])
area = np.abs(np.linalg.det(matrix))
print("坐标系面积的缩放倍数：[{:.9f}]".format(area))

print("\n\n")
print("三维场景(采用了任意维度的通用写法，以三维示例)：")

# 坐标系转换
s = np.array([1, 1, 1])  # 行向量
direction = [np.array([1, 0, 0]), np.array([0, 1, 0]),np.array([1, 1, 1])]
dimension = len(direction)
e_direction = [axis / np.linalg.norm(axis) for axis in direction]  # 单位化
A = np.column_stack(e_direction)
s_new = np.linalg.solve(A, s)  # 线性变换
print("新的s向量为：[{:.9f}, {:.9f},{:.9f}]".format(s_new[0], s_new[1], s_new[2]))

# 坐标系投影
proj_list = [np.dot(s , axis) / np.linalg.norm(axis) for axis in direction]
print("v向量在各轴投影长度",proj_list)

# 坐标系夹角 cos = |(dot)| / |s| * ||
s_norm = np.linalg.norm(s)
d_s = np.linalg.norm(s)
angle_list = []
for axis in direction:
    axis_norm = np.linalg.norm(axis)
    cos_angle = np.dot(s, axis) / (d_s * axis_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angle_list.append(angle)

print("v向量与坐标轴夹角大小：", angle_list)

# 坐标系体积缩放倍数
matrix3 = np.column_stack(direction)
volume = np.abs(np.linalg.det(matrix3))
print("坐标系体积的缩放倍数：[{:.9f}]".format(area))







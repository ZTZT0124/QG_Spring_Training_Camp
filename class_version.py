import numpy as np
import json


class CoordinateSystem:
    def __init__(self, ori_axis):
        self.ori_axis = np.array(ori_axis, dtype=np.float64)
        # 校验基矩阵为方阵（基向量数量=空间维度）
        if self.ori_axis.shape[0] != self.ori_axis.shape[1]:
            raise ValueError(f"基向量必须为方阵，当前形状{self.ori_axis.shape}，基向量数量需等于维度")
        self.dim = self.ori_axis.shape[0]
        # 基矩阵：列向量为基向量（线性代数标准格式）
        self.B = self.ori_axis.T # 转置，设置好原始的基向量矩阵
        # 校验基向量线性无关（可逆）
        det_B = np.linalg.det(self.B)
        if np.isclose(det_B, 0):  # 如果B的行列式的值在误差允许的范围内近似为零，则线性相关无法构成合法坐标系
            raise ValueError(f"基向量线性相关，行列式{det_B:.6f}，无法构成合法坐标系")
        # 预计算逆矩阵，提升坐标转换效率
        self.B_inv = np.linalg.inv(self.B)  # 求解新坐标时使用，B_inv @  即可
    def change_axis(self, obj_axis):
        """切换基准坐标系，更新后所有后续任务均使用新坐标系计算"""
        obj_axis_np = np.array(obj_axis, dtype=np.float64)
        # 校验维度匹配
        if obj_axis_np.shape != (self.dim, self.dim):
            raise ValueError(f"新基向量维度不匹配，当前维度{self.dim}，新基形状{obj_axis_np.shape}")
        new_B = obj_axis_np.T
        # 校验新基可逆
        det_new_B = np.linalg.det(new_B)
        if np.isclose(det_new_B, 0):
            raise ValueError(f"新基向量线性相关，行列式{det_new_B:.6f}，无法构成合法坐标系")
        # 更新坐标系参数
        self.ori_axis = obj_axis_np
        self.B = new_B
        self.B_inv = np.linalg.inv(new_B)
        print(f"坐标系切换成功，新基向量：\n{self.ori_axis}")

    def get_area_scale(self):
        """计算当前坐标系的面积/体积/超体积缩放倍数"""
        return np.abs(np.linalg.det(self.B))

    def calculate_angle(self, vectors):
        """计算每个向量与当前坐标系各基向量的夹角（角度制），零向量标记为NaN"""
        vectors_np = np.array(vectors, dtype=np.float64)
        if vectors_np.shape[1] != self.dim:
            raise ValueError(f"向量维度不匹配，当前维度{self.dim}，向量维度{vectors_np.shape[1]}")
        basis_vectors = self.B.T  # 每行对应一个基向量
        angles = np.zeros((vectors_np.shape[0], self.dim), dtype=np.float64)

        for i, v in enumerate(vectors_np):
            v_norm = np.linalg.norm(v)
            if np.isclose(v_norm, 0):
                angles[i] = np.nan
                continue
            for j, b in enumerate(basis_vectors):
                b_norm = np.linalg.norm(b)
                dot_product = np.dot(v, b)
                # 避免浮点误差导致cos值超出[-1,1]范围
                cos_theta = np.clip(dot_product / (v_norm * b_norm), -1.0, 1.0)
                angles[i, j] = np.degrees(np.arccos(cos_theta))
        return angles

    def calculate_projection(self, vectors):
        """计算每个向量在各基向量上的投影和投影向量（全局坐标）"""
        vectors_np = np.array(vectors, dtype=np.float64)
        if vectors_np.shape[1] != self.dim:
            raise ValueError(f"向量维度不匹配，当前维度{self.dim}，向量维度{vectors_np.shape[1]}")
        basis_vectors = self.B.T
        N = vectors_np.shape[0]
        scalar_proj = np.zeros((N, self.dim), dtype=np.float64)
        vector_proj = np.zeros((N, self.dim, self.dim), dtype=np.float64)

        for i, v in enumerate(vectors_np):
            for j, b in enumerate(basis_vectors):
                b_dot_b = np.dot(b, b)
                if np.isclose(b_dot_b, 0):
                    scalar_proj[i, j] = np.nan
                    vector_proj[i, j] = np.nan
                    continue
                v_dot_b = np.dot(v, b)
                # 标量投影：沿基向量方向的带符号长度
                scalar_proj[i, j] = v_dot_b / np.linalg.norm(b)
                # 投影向量：全局标准基下的坐标
                vector_proj[i, j] = (v_dot_b / b_dot_b) * b
        return scalar_proj, vector_proj

    def get_coordinate_in_current(self, vectors):
        """计算向量在当前坐标系下的坐标"""
        vectors_np = np.array(vectors, dtype=np.float64)
        if vectors_np.shape[1] != self.dim:
            raise ValueError(f"向量维度不匹配，当前维度{self.dim}，向量维度{vectors_np.shape[1]}")
        # 坐标转换公式：v_current = B_inv @ v_global
        return (self.B_inv @ vectors_np.T).T


def process_all_tasks(json_file_path):
    """批量处理JSON文件中的所有任务组"""
    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_groups = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败：{e}")
        return

    # 遍历每个任务组
    for group in all_groups:
        group_name = group.get("group_name", "未知分组")
        print("\n" + "=" * 80)
        print(f"开始处理任务组：{group_name}")
        print("=" * 80)

        # 提取核心数据
        vectors = group.get("vectors", [])
        ori_axis = group.get("ori_axis", [])
        tasks = group.get("tasks", [])

        if not vectors or not ori_axis:
            print(f"任务组{group_name}缺少vectors或ori_axis，跳过")
            continue

        # 转换向量数组
        try:
            vectors_np = np.array(vectors, dtype=np.float64)
        except Exception as e:
            print(f"向量格式转换失败：{e}，跳过该任务组")
            continue

        # 初始化坐标系
        try:
            cs = CoordinateSystem(ori_axis)
        except Exception as e:
            print(f"坐标系初始化失败：{e}，跳过该任务组")
            continue

        # 打印初始信息
        print(f"坐标系初始化成功，空间维度：{cs.dim}")
        print(f"初始基向量：\n{cs.ori_axis}")
        print(f"初始缩放倍数：{cs.get_area_scale():.4f}")
        print(f"待处理向量数：{len(vectors_np)} | 待处理任务数：{len(tasks)}")
        print("-" * 60)

        # 按顺序执行所有任务
        for task_idx, task in enumerate(tasks):
            task_type = task.get("type", "未知类型")
            print(f"\n执行第{task_idx + 1}个任务：{task_type}")

            try:
                if task_type == "area":
                    scale = cs.get_area_scale()
                    print(f"当前坐标系缩放倍数：{scale:.6f}")

                elif task_type == "axis_angle":
                    angles = cs.calculate_angle(vectors_np)
                    print("向量与各基向量的夹角（角度制）：")
                    for i, angle in enumerate(angles):
                        print(f"  向量{i + 1} {vectors[i]} → {np.round(angle, 4)}")

                elif task_type == "axis_projection":
                    scalar_proj, vector_proj = cs.calculate_projection(vectors_np)
                    print("向量在各基向量上的投影：")
                    for i in range(len(vectors_np)):
                        print(f"  向量{i + 1} {vectors[i]}：")
                        print(f"    标量投影：{np.round(scalar_proj[i], 4)}")
                        print(f"    投影向量（全局坐标）：\n{np.round(vector_proj[i], 4)}")

                elif task_type == "change_axis":
                    obj_axis = task.get("obj_axis", [])
                    if not obj_axis:
                        print(f"缺少obj_axis参数，跳过该任务")
                        continue
                    cs.change_axis(obj_axis)

                else:
                    print(f"未知任务类型：{task_type}，跳过")

            except Exception as e:
                print(f"执行任务失败：{e}")
                continue

        # 打印任务组最终结果
        print("\n" + "-" * 60)
        print(f"任务组{group_name}执行完毕")
        print(f"最终基向量：\n{cs.ori_axis}")
        print(f"最终缩放倍数：{cs.get_area_scale():.4f}")
        print("所有向量在最终坐标系下的坐标：")
        final_coords = cs.get_coordinate_in_current(vectors_np)
        for i, coord in enumerate(final_coords):
            print(f"  向量{i + 1} {vectors[i]} → {np.round(coord, 4)}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    process_all_tasks("data.json")
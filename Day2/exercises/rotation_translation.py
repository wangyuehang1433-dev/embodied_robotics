
import numpy as np

# 一个点在局部坐标系中的坐标
p_local = np.array([0.0, 0.4])

# 局部坐标系绕Y轴旋转90度
theta = np.deg2rad(90)

# X-Z平面中的旋转矩阵
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]
])

# 局部原点在世界坐标系中的位置
t = np.array([0.2, 0.2])

# 坐标变换：先旋转，再平移
p_world = R @ p_local + t

print("局部坐标:", p_local)
print("旋转矩阵:")
print(R)
print("平移向量:", t)
print("世界坐标:", p_world)

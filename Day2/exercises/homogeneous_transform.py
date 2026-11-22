
import numpy as np

def transform(theta, tx, tz):
    """构建X-Z平面中的3×3齐次变换矩阵"""
    c = np.cos(theta)
    s = np.sin(theta)

    T = np.array([
        [c,  s, tx],
        [-s, c, tz],
        [0,  0, 1]
    ])

    return T


# 实验1：单个坐标系的旋转和平移
theta = np.deg2rad(90)

T = transform(theta, 0.2, 0.2)

p_local = np.array([0, 0.4, 1])

p_world = T @ p_local

print("实验1：齐次变换")
print("T =")
print(T)
print("世界坐标:", p_world[:2])


# 实验2：双关节机械臂正运动学
q1 = 0.6
q2 = -0.8

L1 = 0.5
L2 = 0.4
z0 = 0.2

# 世界 -> 基座
T_WB = transform(0, 0, z0)

# 基座 -> 第一连杆坐标系
T_B1 = transform(q1, 0, 0)

# 第一连杆 -> 第二连杆坐标系
T12 = transform(q2, 0, L1)_

# 第二连杆 -> 末端
T_2E = transform(0, 0, L2)

# 逐级组合变换
T_WE = T_WB @ T_B1 @ T_12 @ T_2E

print("\n实验2：双关节机械臂")
print("T_WE =")
print(T_WE)

print("末端X坐标:", T_WE[0, 2])
print("末端Z坐标:", T_WE[1, 2])

# 用之前的正运动学公式验证
x_fk = L1*np.sin(q1) + L2*np.sin(q1 + q2)
z_fk = z0 + L1*np.cos(q1) + L2*np.cos(q1 + q2)

print("\n正运动学验证:")
print("FK X:", x_fk)
print("FK Z:", z_fk)

print("是否一致:", np.allclose(
    [T_WE[0, 2], T_WE[1, 2]],
    [x_fk, z_fk]
))


import numpy as np

# 连杆长度
L1 = 0.5
L2 = 0.4

# 正运动学
def forward_kinematics(q1, q2):

    x = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)

    z = 0.2 + L1 * np.cos(q1) + L2 * np.cos(q1 + q2)

    return np.array([x, z])


# 雅可比矩阵
def jacobian(q1, q2):

    J = np.array([
        [
            L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
            L2 * np.cos(q1 + q2)
        ],
        [
            -L1 * np.sin(q1) - L2 * np.sin(q1 + q2),
            -L2 * np.sin(q1 + q2)
        ]
    ])

    return J


# 当前关节角度
q = np.array([0.6, -0.8])

# 当前关节角速度
q_dot = np.array([0.5, 0.2])

# 计算雅可比矩阵
J = jacobian(q[0], q[1])

# 计算末端速度
p_dot = J @ q_dot

print("关节角度:", q)
print("关节角速度:", q_dot)

print("\n雅可比矩阵:")
print(J)

print("\n末端位置:")
print(forward_kinematics(q[0], q[1]))

print("\n末端速度 [vx, vz]:")
print(p_dot)

# 使用有限差分验证雅可比矩阵

eps = 1e-6

J_numeric = np.zeros((2, 2))

p0 = forward_kinematics(q[0], q[1])

for i in range(2):

    q_new = q.copy()

    q_new[i] += eps

    p_new = forward_kinematics(
        q_new[0],
        q_new[1]
    )

    J_numeric[:, i] = (p_new - p0) / eps


print("\n解析雅可比矩阵:")
print(J)

print("\n数值雅可比矩阵:")
print(J_numeric)

print("\n两者最大误差:")
print(np.max(np.abs(J - J_numeric)))


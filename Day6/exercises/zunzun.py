
import numpy as np

# =====================================
# 1. 机械臂参数
# =====================================

L1 = 0.5
L2 = 0.4

def jacobian(q1, q2):
    return np.array([
        [
            L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
            L2 * np.cos(q1 + q2)
        ],
        [
            -L1 * np.sin(q1) - L2 * np.sin(q1 + q2),
            -L2 * np.sin(q1 + q2)
        ]
    ])

# =====================================
# 2. 固定机械臂姿态
# =====================================

q1 = np.deg2rad(0)
q2 = np.deg2rad(-5)

J = jacobian(q1, q2)

# 希望末端沿 Z 方向运动
v_desired = np.array([0.0, 0.1])

# 不同阻尼系数
damping_values = [
    0.001,
    0.01,
    0.05,
    0.1,
    0.2
]

print("===== Day6 阻尼系数实验 =====")

print("机械臂姿态: q1=0°, q2=-5°")
print("期望末端速度:", v_desired)

# =====================================
# 3. 逐个测试阻尼
# =====================================

for damping in damping_values:

    q_dot = J.T @ np.linalg.solve(
        J @ J.T + damping**2 * np.eye(2),
        v_desired
    )

    # 计算运动学预测的末端速度
    v_achieved = J @ q_dot

    # 关节速度范数
    q_speed = np.linalg.norm(q_dot)

    # 末端速度误差
    v_error = np.linalg.norm(
        v_desired - v_achieved
    )

    print(f"\n阻尼 λ = {damping}")

    print(
        "关节速度:",
        np.round(q_dot, 4)
    )

    print(
        "关节速度范数:",
        round(q_speed, 4),
        "rad/s"
    )

    print(
        "预测末端速度:",
        np.round(v_achieved, 5),
        "m/s"
    )

    print(
        "末端速度误差:",
        round(v_error, 5),
        "m/s"
    )

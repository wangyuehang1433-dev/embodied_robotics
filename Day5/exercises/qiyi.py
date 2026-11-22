
import numpy as np

L1 = 0.5
L2 = 0.4

damping = 0.05

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

q1 = 0.0

angles_deg = [-90, -45, -20, -5, -1]

# 期望末端沿 Z 方向运动
# q1=0 且 q2 接近 0 时，Z 方向正是困难方向
v_desired = np.array([0.0, 0.1])

print("===== 普通逆雅可比 vs DLS =====")

for angle in angles_deg:
    q2 = np.deg2rad(angle)

    J = jacobian(q1, q2)

    # 方法 A：普通逆雅可比
    qdot_inverse = np.linalg.solve(
        J, v_desired
    )

    # 方法 B：阻尼最小二乘法
    qdot_dls = J.T @ np.linalg.solve(
        J @ J.T + damping**2 * np.eye(2),
        v_desired
    )

    print(f"\nq2 = {angle}°")

    print(
        "普通逆雅可比关节速度:",
        np.round(qdot_inverse, 3)
    )

    print(
        "DLS 关节速度:",
        np.round(qdot_dls, 3)
    )

    print(
        "普通逆雅可比速度范数:",
        round(np.linalg.norm(qdot_inverse), 3)
    )

    print(
        "DLS 速度范数:",
        round(np.linalg.norm(qdot_dls), 3)
    )

    print(
        "DLS 实际可实现末端速度:",
        np.round(J @ qdot_dls, 4)
    )

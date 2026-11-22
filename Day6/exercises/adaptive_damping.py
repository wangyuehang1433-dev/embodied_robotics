
import numpy as np

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

def adaptive_damping(J):
    singular_values = np.linalg.svd(
        J, compute_uv=False
    )

    sigma_min = np.min(singular_values)

    sigma_threshold = 0.1
    lambda_max = 0.1

    if sigma_min >= sigma_threshold:
        damping = 0.0
    else:
        damping = lambda_max * (
            1 - sigma_min / sigma_threshold
        )

    return damping, sigma_min


q1 = 0.0

angles_deg = [-90, -45, -20, -5, -1, 0]

v_desired = np.array([0.0, 0.1])

print("===== Day6 自适应阻尼实验 =====")

for angle in angles_deg:
    q2 = np.deg2rad(angle)

    J = jacobian(q1, q2)

    damping, sigma_min = adaptive_damping(J)

    # 通过最小二乘求解正规方程
    # 在 damping = 0 且 J 奇异时，
    # 改用伪逆，避免矩阵求解失败
    if damping == 0.0:
        q_dot = np.linalg.pinv(J) @ v_desired
    else:
        q_dot = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(2),
            v_desired
        )

    v_achieved = J @ q_dot

    q_speed = np.linalg.norm(q_dot)

    v_error = np.linalg.norm(
        v_desired - v_achieved
    )

    print(f"\nq2 = {angle}°")
    print(f"最小奇异值: {sigma_min:.5f}")
    print(f"自适应阻尼: {damping:.5f}")
    print(f"关节速度范数: {q_speed:.4f} rad/s")
    print(f"末端速度误差: {v_error:.5f} m/s")

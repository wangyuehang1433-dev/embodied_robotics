
import numpy as np

L1 = 0.5
L2 = 0.4
z0 = 0.2


# 1. 正运动学：角度 -> 位置
def forward_kinematics(q):
    q1, q2 = q

    x = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    z = z0 + L1 * np.cos(q1) + L2 * np.cos(q1 + q2)

    return np.array([x, z])


# 2. 雅可比矩阵：角速度 -> 末端速度
def jacobian(q):
    q1, q2 = q

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


# 3. 当前关节角度
q = np.array([0.6, -0.8])

# 4. 当前关节角速度（rad/s）
q_dot = np.array([0.2, -0.1])

# 5. 使用雅可比矩阵计算末端速度
J = jacobian(q)
v_jacobian = J @ q_dot

print("雅可比矩阵:")
print(J)

print("\n关节角速度:")
print(q_dot)

print("\n雅可比计算的末端速度:")
print(v_jacobian)


# 6. 使用有限差分验证
dt = 0.0001

p_before = forward_kinematics(q)

# 假设在极短时间内关节保持当前角速度
q_after = q + q_dot * dt

p_after = forward_kinematics(q_after)

# 位移 / 时间 = 平均速度
v_numerical = (p_after - p_before) / dt

print("\n有限差分计算的末端速度:")
print(v_numerical)

print("\n两种方法是否接近:")
print(np.allclose(
    v_jacobian,
    v_numerical,
    atol=1e-4
))

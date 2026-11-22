
import numpy as np

L1 = 0.5
L2 = 0.4
z0 = 0.2


def forward_kinematics(q):
    q1, q2 = q

    x = L1*np.sin(q1) + L2*np.sin(q1+q2)
    z = z0 + L1*np.cos(q1) + L2*np.cos(q1+q2)

    return np.array([x, z])


def jacobian(q):
    q1, q2 = q

    return np.array([
        [
            L1*np.cos(q1) + L2*np.cos(q1+q2),
            L2*np.cos(q1+q2)
        ],
        [
            -L1*np.sin(q1) - L2*np.sin(q1+q2),
            -L2*np.sin(q1+q2)
        ]
    ])


# 初始关节角度
q = np.array([0.6, -0.8])

# 目标末端位置
target = np.array([0.4, 0.8])

# 控制参数
K = 2.0
dt = 0.01
max_steps = 1000

for step in range(max_steps):

    # 1. 计算当前末端位置
    current = forward_kinematics(q)

    # 2. 计算位置误差
    error = target - current

    # 3. 判断是否到达目标
    if np.linalg.norm(error) < 0.001:
        print("到达目标！")
        break

    # 4. 根据误差生成期望末端速度
    v_desired = K * error

    # 5. 计算关节角速度
    J = jacobian(q)
    q_dot = np.linalg.pinv(J) @ v_desired

    # 6. 更新关节角度
    q = q + q_dot * dt

    if step % 100 == 0:
        print(
            f"step={step}, "
            f"position={current}, "
            f"error={np.linalg.norm(error):.6f}"
        )

else:
    print("达到最大迭代次数，尚未满足误差阈值")

print("\n最终关节角度:", q)
print("最终末端位置:", forward_kinematics(q))
print("目标末端位置:", target)
print("最终误差:", np.linalg.norm(target - forward_kinematics(q)))

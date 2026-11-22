
import numpy as np

L1 = 0.5
L2 = 0.4

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


# 当前关节角度
q = np.array([0.6, -0.8])

# 希望末端向右运动0.1 m/s
v_target = np.array([0.1, 0.0])

# 计算雅可比矩阵
J = jacobian(q)

# 计算雅可比伪逆
J_pinv = np.linalg.pinv(J)

# 计算需要的关节角速度
q_dot = J_pinv @ v_target

# 验证：重新计算末端速度
v_actual = J @ q_dot

print("雅可比矩阵:")
print(J)

print("\n雅可比伪逆:")
print(J_pinv)

print("\n目标末端速度:")
print(v_target)

print("\n计算得到的关节角速度:")
print(q_dot)

print("\n重新计算的末端速度:")
print(v_actual)

print("\n是否达到目标速度:")
print(np.allclose(v_target, v_actual))

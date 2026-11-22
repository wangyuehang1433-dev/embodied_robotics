
import numpy as np

# 连杆长度
L1 = 0.5
L2 = 0.4

# 基座高度
Z0 = 0.2


def forward_kinematics(q1, q2):

    # 计算末端X坐标
    x = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)

    # 计算末端Z坐标
    z = Z0 + L1 * np.cos(q1) + L2 * np.cos(q1 + q2)

    return np.array([x, z])


# 测试关节角度
q1 = 0.6
q2 = -0.8

position = forward_kinematics(q1, q2)

print("关节角度:", q1, q2)
print("末端X坐标:", position[0])
print("末端Z坐标:", position[1])

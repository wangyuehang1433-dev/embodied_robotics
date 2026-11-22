
import numpy as np
import mujoco

XML = """
<mujoco model="two_link_jacobian">
    <compiler angle="radian"/>

    <worldbody>
        <body name="link1" pos="0 0 0.2">
            <joint name="joint1" type="hinge"
                   axis="0 1 0"/>

            <geom type="capsule"
                  fromto="0 0 0 0 0 0.5"
                  size="0.035" mass="1"/>

            <body name="link2" pos="0 0 0.5">
                <joint name="joint2" type="hinge"
                       axis="0 1 0"/>

                <geom type="capsule"
                      fromto="0 0 0 0 0 0.4"
                      size="0.025" mass="0.5"/>

                <site name="end_effector"
                      pos="0 0 0.4"
                      size="0.04"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# 1. 设置关节角度和角速度
q = np.array([0.6, -0.8])
q_dot = np.array([0.2, -0.1])

data.qpos[:] = q
data.qvel[:] = q_dot

mujoco.mj_forward(model, data)

# 2. 自己推导的雅可比矩阵
q1, q2 = q

J_ours = np.array([
    [
        0.5*np.cos(q1) + 0.4*np.cos(q1+q2),
        0.4*np.cos(q1+q2)
    ],
    [
        -0.5*np.sin(q1) - 0.4*np.sin(q1+q2),
        -0.4*np.sin(q1+q2)
    ]
])

v_ours = J_ours @ q_dot

# 3. MuJoCo内置雅可比矩阵
site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "end_effector"
)

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

# 取出X、Z方向
J_mujoco = jacp[[0, 2], :]

v_mujoco = J_mujoco @ data.qvel

# 4. 对比结果
print("自己推导的雅可比矩阵:")
print(J_ours)

print("\nMuJoCo计算的雅可比矩阵:")
print(J_mujoco)

print("\n自己计算的末端速度:")
print(v_ours)

print("\nMuJoCo计算的末端速度:")
print(v_mujoco)

print("\n雅可比矩阵是否一致:")
print(np.allclose(J_ours, J_mujoco))

print("\n末端速度是否一致:")
print(np.allclose(v_ours, v_mujoco))

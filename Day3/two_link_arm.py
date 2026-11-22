
import time
import numpy as np
import mujoco
import mujoco.viewer

# ==================================
# 1. 创建双关节机械臂模型
# ==================================

XML = """
<mujoco model="ik_tracking">
    <compiler angle="radian"/>
    <option timestep="0.002" gravity="0 0 0"/>

    <worldbody>
        <geom type="plane" size="2 2 0.1"/>

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
                      size="0.04"
                      rgba="0 1 0 1"/>
            </body>
        </body>

        <!-- 红色目标点，世界坐标系 -->
        <site name="target"
              pos="0.4 0 0.8"
              size="0.045"
              rgba="1 0 0 1"/>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# ==================================
# 2. 初始化
# ==================================

q = np.array([0.6, -0.8], dtype=float)
target = np.array([0.4, 0.8])

K = 2.0
dt = 0.01
max_steps = 1000

data.qpos[:] = q
mujoco.mj_forward(model, data)

ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "end_effector"
)

# ==================================
# 3. 逆运动学控制循环
# ==================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    for step in range(max_steps):

        if not viewer.is_running():
            break

        # 读取MuJoCo中的末端位置
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[ee_id].copy()

        current = np.array([ee_pos[0], ee_pos[2]])

        # 计算位置误差
        error = target - current
        error_norm = np.linalg.norm(error)

        if step % 100 == 0:
            print(
                f"step={step}, "
                f"position={current}, "
                f"error={error_norm:.6f}"
            )

        if error_norm < 0.001:
            print("末端到达目标！")
            break

        # 使用MuJoCo内置雅可比矩阵
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model, data, jacp, jacr, ee_id
        )

        # 提取X、Z方向
        J = jacp[[0, 2], :]

        # 期望末端速度
        v_desired = K * error

        # 雅可比伪逆计算关节角速度
        q_dot = np.linalg.pinv(J) @ v_desired

        # 更新关节角度
        q = q + q_dot * dt

        data.qpos[:] = q
        data.qvel[:] = q_dot

        # 更新MuJoCo运动学
        mujoco.mj_forward(model, data)

        # 刷新窗口
        viewer.sync()

        # 放慢速度，方便观察
        time.sleep(dt)

    print("\n最终关节角度:", data.qpos.copy())
    print("最终末端位置:", data.site_xpos[ee_id].copy())
    print("目标位置:", target)

    # 到达目标后，继续显示窗口
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)

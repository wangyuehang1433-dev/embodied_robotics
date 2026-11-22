
import numpy as np
import mujoco
import mujoco.viewer

# ==========================================
# 1. 初始角度和目标位置
# ==========================================

# 希望 Reset 后恢复的关节角度（弧度）
INITIAL_Q1 = -1.2
INITIAL_Q2 = 0

# 末端目标位置（世界坐标系 XZ 平面）
target_pos = np.array([0.4, 0.8])

# ==========================================
# 2. MuJoCo 模型
# ==========================================

# 关键：
# joint ref 表示模型默认状态下的关节坐标
# 编译时会建立对应的参考几何配置
#
# 因为我们希望：
# q=[0,0] 时两根连杆竖直
# q=[0.6,-0.8] 时为初始倾斜姿态
#
# 所以在 XML 中对连杆几何做相应旋转。

XML = f"""
<mujoco model="online_ik_pd">
    <compiler angle="radian"/>
    <option timestep="0.002" gravity="0 0 0"/>

    <worldbody>
        <geom type="plane" size="2 2 0.1"/>

        <!-- 世界坐标系Z轴参考线 -->
        <geom name="z_reference"
              type="capsule"
              fromto="0 0 0.2 0 0 1.3"
              size="0.008"
              rgba="0 1 0 0.4"
              contype="0"
              conaffinity="0"/>

        <!-- 世界坐标系X轴参考线 -->
        <geom name="x_reference"
              type="capsule"
              fromto="-0.2 0 0.2 1.0 0 0.2"
              size="0.008"
              rgba="1 0 0 0.4"
              contype="0"
              conaffinity="0"/>

        <body name="link1"
              pos="0 0 0.2"
              euler="0 {INITIAL_Q1} 0">

            <joint name="joint1"
                   type="hinge"
                   axis="0 1 0"
                   ref="{INITIAL_Q1}"/>

            <geom type="capsule"
                  fromto="0 0 0 0 0 0.5"
                  size="0.035"
                  mass="1"
                  rgba="0.2 0.4 0.9 1"/>

            <body name="link2"
                  pos="0 0 0.5"
                  euler="0 {INITIAL_Q2} 0">

                <joint name="joint2"
                       type="hinge"
                       axis="0 1 0"
                       ref="{INITIAL_Q2}"/>

                <geom type="capsule"
                      fromto="0 0 0 0 0 0.4"
                      size="0.025"
                      mass="0.5"
                      rgba="1 0.6 0.2 1"/>

                <!-- 绿色末端 -->
                <site name="end_effector"
                      pos="0 0 0.4"
                      size="0.025"
                      rgba="0 1 0 1"/>
            </body>
        </body>

        <!-- 红色目标 -->
        <site name="target"
              pos="0.4 0 0.8"
              size="0.025"
              rgba="1 0 0 1"/>
    </worldbody>

    <actuator>
        <motor name="motor1"
               joint="joint1"
               ctrlrange="-30 30"/>

        <motor name="motor2"
               joint="joint2"
               ctrlrange="-20 20"/>
    </actuator>
</mujoco>
"""

# ==========================================
# 3. 初始化
# ==========================================

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "end_effector"
)

print("模型默认关节角度:", model.qpos0)
print("当前关节角度:", data.qpos[:2])
print("初始末端位置:", data.site_xpos[ee_id])

# ==========================================
# 4. 控制参数
# ==========================================

K_ik = 2.0
damping = 0.05
max_qdot = 1.0

kp = np.array([60.0, 40.0])
kd = np.array([20.0, 14.0])

torque_min = np.array([-30.0, -20.0])
torque_max = np.array([30.0, 20.0])

dt = model.opt.timestep

# ==========================================
# 5. 在线IK + PD控制器
# ==========================================

last_print_time = -1.0


def controller(model, data):
    global last_print_time

    # 当前关节状态
    q = data.qpos[:2].copy()
    q_dot = data.qvel[:2].copy()

    # 当前末端位置
    ee_pos = data.site_xpos[ee_id][[0, 2]].copy()

    # 末端位置误差
    error = target_pos - ee_pos

    # 雅可比矩阵
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        ee_id
    )

    J = jacp[[0, 2], :2]

    # 期望末端速度
    v_desired = K_ik * error

    # DLS阻尼最小二乘
    A = J @ J.T + damping**2 * np.eye(2)

    q_dot_desired = J.T @ np.linalg.solve(
        A,
        v_desired
    )

    q_dot_desired = np.clip(
        q_dot_desired,
        -max_qdot,
        max_qdot
    )

    # 下一小步目标关节角度
    q_target_next = q + q_dot_desired * dt

    # PD控制
    torque = (
        kp * (q_target_next - q)
        + kd * (q_dot_desired - q_dot)
    )

    data.ctrl[:] = np.clip(
        torque,
        torque_min,
        torque_max
    )

    # 每隔1秒打印
    if data.time < last_print_time:
        last_print_time = -1.0

    if data.time - last_print_time >= 1.0:
        last_print_time = data.time

        print("\n==========================")
        print(f"仿真时间: {data.time:.2f} s")
        print("当前关节角度:", q)
        print("末端位置:", ee_pos)
        print("目标位置:", target_pos)
        print(
            "末端误差(mm):",
            np.linalg.norm(error) * 1000
        )
        print("期望关节速度:", q_dot_desired)
        print("实际关节速度:", q_dot)
        print("==========================")


# ==========================================
# 6. 启动 Viewer
# ==========================================

mujoco.set_mjcb_control(controller)

print("\n在线IK + PD实验")
print("Run：开始")
print("Pause：暂停")
print("Reset：恢复 [0.6, -0.8]")
print("绿色参考线：Z轴")
print("红色参考线：X轴")

try:
    mujoco.viewer.launch(model, data)
finally:
    mujoco.set_mjcb_control(None)

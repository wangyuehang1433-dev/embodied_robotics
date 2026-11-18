
import time
import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco model="two_link_arm">
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
                      pos="0 0 0.4" size="0.04"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <motor name="motor1" joint="joint1"
               ctrlrange="-30 30"/>
        <motor name="motor2" joint="joint2"
               ctrlrange="-20 20"/>
    </actuator>·
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "end_effector"
)

data.qpos[:2] = [0.6, 0.8]

mujoco.mj_forward(model, data)

print("MuJoCo末端位置:", data.site_xpos[site_id])
print("关节数量:", model.njnt)
print("执行器数量:", model.nu)

kp = np.array([60.0, 40.0])
kd = np.array([12.0, 8.0])

target = np.array([0.6,0])

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        q = data.qpos[:2].copy()
        dq = data.qvel[:2].copy()

        error = target - q

        torque = kp * error - kd * dq

        data.ctrl[:] = np.clip(
            torque,
            [-30, -20],
            [30, 20]
        )

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)


print("最终关节角度:", data.qpos[:2])
print("目标关节角度:", target)

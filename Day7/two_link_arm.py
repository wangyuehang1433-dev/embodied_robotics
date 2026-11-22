
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import csv
from pathlib import Path


# ==========================================
# 1. 参数
# ==========================================

# 初始关节角度
# 先使用非奇异初始姿态，便于与 Day 5/6 对比
INITIAL_Q1 = -0.6
INITIAL_Q2 = -0.8

# 最终目标位置 [x, z]
goal_pos = np.array([0.4, 0.8], dtype=float)

# 五次多项式轨迹运动时间
T = 4.0

# Day 7：任务空间 PD 参数
# Kp_task 单位近似为 N/m
# Kd_task 单位近似为 N·s/m
Kp_task = np.diag([10.0, 10.0])
Kd_task = np.diag([20.0, 20.0])

# 关节力矩限制
torque_min = np.array([-30.0, -20.0])
torque_max = np.array([30.0, 20.0])

# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. MuJoCo XML
# ==========================================

XML = f"""
<mujoco model="day7_task_space_pd">

    <compiler angle="radian"/>

    <option timestep="0.002"
            gravity="0 0 0"/>

    <worldbody>

        <geom type="plane"
              size="2 2 0.1"/>

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

                <site name="end_effector"
                      pos="0 0 0.4"
                      size="0.025"
                      rgba="0 1 0 1"/>

            </body>
        </body>

        <!-- 红色：最终目标点 -->
        <site name="goal"
              pos="0.4 0 0.8"
              size="0.025"
              rgba="1 0 0 1"/>

    </worldbody>

    <actuator>

        <motor name="motor1"
               joint="joint1"
               ctrlrange="-30 30"
               ctrllimited="true"/>

        <motor name="motor2"
               joint="joint2"
               ctrlrange="-20 20"
               ctrllimited="true"/>

    </actuator>

</mujoco>
"""


# ==========================================
# 3. 创建 MuJoCo 模型
# ==========================================

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

dt = model.opt.timestep

ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "end_effector"
)

# 读取真实初始末端位置
start_pos = data.site_xpos[ee_id][[0, 2]].copy()

print("初始关节角度:", data.qpos[:2])
print("初始末端位置:", start_pos)
print("最终目标位置:", goal_pos)
print("仿真时间步长:", dt)


# ==========================================
# 4. 五次多项式轨迹规划
# ==========================================

def trajectory(t):

    tau = np.clip(t / T, 0.0, 1.0)

    s = (
        10 * tau**3
        - 15 * tau**4
        + 6 * tau**5
    )

    return start_pos + s * (goal_pos - start_pos)


def trajectory_velocity(t):

    if 0.0 <= t < T:

        tau = t / T

        s_dot = (
            30 * tau**2
            - 60 * tau**3
            + 30 * tau**4
        ) / T

        return s_dot * (goal_pos - start_pos)

    return np.zeros(2)


# ==========================================
# 5. 控制器状态与实验数据
# ==========================================

last_time = float(data.time)
last_print_time = -1.0
last_record_time = -1.0

# 每条记录：
# [
#   time,
#   target_x, target_z,
#   actual_x, actual_z,
#   error_mm,
#   torque_1, torque_2,
#   raw_torque_1, raw_torque_2
# ]
records = []


# ==========================================
# 6. Day 7：任务空间 PD 控制器
# ==========================================

def controller(model, data):

    global last_time
    global last_print_time
    global last_record_time

    current_time = float(data.time)

    # --------------------------------------
    # A. 检测 Viewer Reset
    # --------------------------------------

    if current_time < last_time - 1e-9:

        last_print_time = -1.0
        last_record_time = -1.0

        records.clear()

        print(
            "\n检测到 Reset："
            "实验数据已重置"
        )

    last_time = current_time

    # --------------------------------------
    # B. 读取当前状态
    # --------------------------------------

    q_dot = data.qvel[:2].copy()

    ee_pos = data.site_xpos[ee_id][[0, 2]].copy()

    # --------------------------------------
    # C. 计算期望末端轨迹
    # --------------------------------------

    target_pos = trajectory(current_time)
    target_vel = trajectory_velocity(current_time)

    # --------------------------------------
    # D. 计算雅可比矩阵
    # --------------------------------------

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        ee_id
    )

    # 只取 X、Z 两个方向
    J = jacp[[0, 2], :2]

    # --------------------------------------
    # E. 计算末端实际速度
    # --------------------------------------

    ee_vel = J @ q_dot

    # --------------------------------------
    # F. 任务空间位置与速度误差
    # --------------------------------------

    pos_error = target_pos - ee_pos
    vel_error = target_vel - ee_vel

    error_mm = np.linalg.norm(pos_error) * 1000.0

    # --------------------------------------
    # G. 任务空间 PD
    # --------------------------------------

    # 虚拟末端控制力
    F = (
        Kp_task @ pos_error
        + Kd_task @ vel_error
    )

    # --------------------------------------
    # H. 雅可比转置映射到关节力矩
    # --------------------------------------

    torque = J.T @ F

    # --------------------------------------
    # I. 关节力矩限幅
    # --------------------------------------

    torque_clipped = np.clip(
        torque,
        torque_min,
        torque_max
    )

    data.ctrl[:] = torque_clipped

    # --------------------------------------
    # J. 记录实验数据
    # --------------------------------------

    if current_time > last_record_time + 1e-9:

        records.append([
            current_time,
            float(target_pos[0]),
            float(target_pos[1]),
            float(ee_pos[0]),
            float(ee_pos[1]),
            float(error_mm),
            float(torque_clipped[0]),
            float(torque_clipped[1]),
            float(torque[0]),
            float(torque[1])
        ])

        last_record_time = current_time

    # --------------------------------------
    # K. 每 0.5 秒打印一次
    # --------------------------------------

    if current_time - last_print_time >= 0.5:

        last_print_time = current_time

        print(
            f"t={current_time:.2f}s | "
            f"target={np.round(target_pos, 4)} | "
            f"actual={np.round(ee_pos, 4)} | "
            f"error={error_mm:.2f} mm | "
            f"F={np.round(F, 2)} | "
            f"torque={np.round(torque_clipped, 2)}"
        )


# ==========================================
# 7. 误差指标计算
# ==========================================

def calculate_metrics(errors):

    max_error = np.max(errors)

    mean_error = np.mean(errors)

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    return max_error, mean_error, rmse


# ==========================================
# 8. 保存数据并绘图
# ==========================================

def analyze_and_plot():

    if len(records) < 2:

        print("\n数据不足，无法绘图")
        return

    data_array = np.array(records)

    times = data_array[:, 0]

    target_x = data_array[:, 1]
    target_z = data_array[:, 2]

    actual_x = data_array[:, 3]
    actual_z = data_array[:, 4]

    errors = data_array[:, 5]

    torque_1 = data_array[:, 6]
    torque_2 = data_array[:, 7]

    raw_torque_1 = data_array[:, 8]
    raw_torque_2 = data_array[:, 9]

    # --------------------------------------
    # A. 保存 CSV
    # --------------------------------------

    csv_path = OUTPUT_DIR / "day7_tracking_data.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "time",
            "target_x",
            "target_z",
            "actual_x",
            "actual_z",
            "error_mm",
            "torque_1",
            "torque_2",
            "raw_torque_1",
            "raw_torque_2"
        ])

        writer.writerows(records)

    # --------------------------------------
    # B. 全程跟踪误差
    # --------------------------------------

    max_error, mean_error, rmse = (
        calculate_metrics(errors)
    )

    print("\n========== 全程跟踪误差 ==========")

    print(f"最大误差: {max_error:.3f} mm")
    print(f"平均误差: {mean_error:.3f} mm")
    print(f"RMSE:     {rmse:.3f} mm")

    # --------------------------------------
    # C. 0～4 秒运动阶段误差
    # --------------------------------------

    motion_mask = times <= T
    motion_errors = errors[motion_mask]
    motion_times = times[motion_mask]

    if len(motion_errors) > 0:

        motion_max, motion_mean, motion_rmse = (
            calculate_metrics(motion_errors)
        )

        print("\n========== 运动阶段误差 ==========")

        print(
            f"实际统计区间: "
            f"{motion_times[0]:.3f} ~ "
            f"{motion_times[-1]:.3f} s"
        )

        if motion_times[-1] < T - dt:

            print(
                "注意：尚未完整记录到 4 秒，"
                "当前指标不能作为完整 0～4 秒结果。"
            )

        print(f"最大误差: {motion_max:.3f} mm")
        print(f"平均误差: {motion_mean:.3f} mm")
        print(f"RMSE:     {motion_rmse:.3f} mm")

    # --------------------------------------
    # D. 绘制末端轨迹对比图
    # --------------------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        target_x,
        target_z,
        "b--",
        linewidth=2,
        label="Desired trajectory"
    )

    plt.plot(
        actual_x,
        actual_z,
        "r-",
        linewidth=1.5,
        label="Actual trajectory"
    )

    plt.scatter(
        start_pos[0],
        start_pos[1],
        color="green",
        s=70,
        label="Start",
        zorder=5
    )

    plt.scatter(
        goal_pos[0],
        goal_pos[1],
        color="black",
        s=70,
        label="Goal",
        zorder=5
    )

    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")

    plt.title(
        "Day 7: Task-Space PD Trajectory Tracking"
    )

    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    trajectory_path = (
        OUTPUT_DIR / "day7_trajectory_tracking.png"
    )

    plt.savefig(
        trajectory_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------
    # E. 绘制末端位置误差曲线
    # --------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        errors,
        color="blue",
        linewidth=1.5,
        label="Tracking error"
    )

    plt.axvline(
        x=T,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Trajectory end (4s)"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (mm)")

    plt.title(
        "Day 7: End-Effector Tracking Error"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    error_path = (
        OUTPUT_DIR / "day7_tracking_error.png"
    )

    plt.savefig(
        error_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------
    # F. 绘制关节力矩曲线
    # --------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        torque_1,
        label="Joint 1 torque",
        linewidth=1.5
    )

    plt.plot(
        times,
        torque_2,
        label="Joint 2 torque",
        linewidth=1.5
    )

    # 两个关节的力矩限制不同
    plt.axhline(
        y=30.0,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Joint 1 limit (+/-30)"
    )

    plt.axhline(
        y=-30.0,
        color="gray",
        linestyle="--",
        alpha=0.5
    )

    plt.axhline(
        y=20.0,
        color="orange",
        linestyle=":",
        alpha=0.6,
        label="Joint 2 limit (+/-20)"
    )

    plt.axhline(
        y=-20.0,
        color="orange",
        linestyle=":",
        alpha=0.6
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N m)")

    plt.title("Day 7: Joint Control Torques")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    torque_path = (
        OUTPUT_DIR / "day7_joint_torque.png"
    )

    plt.savefig(
        torque_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------
    # G. 力矩饱和统计
    # --------------------------------------

    sat_1 = np.mean(
        np.abs(raw_torque_1) > 30.0 + 1e-9
    ) * 100.0

    sat_2 = np.mean(
        np.abs(raw_torque_2) > 20.0 + 1e-9
    ) * 100.0

    print("\n========== 力矩饱和统计 ==========")

    print(f"关节 1 饱和比例: {sat_1:.2f}%")
    print(f"关节 2 饱和比例: {sat_2:.2f}%")

    # --------------------------------------
    # H. 打印保存结果
    # --------------------------------------

    print("\n========== 数据保存完成 ==========")

    print(f"记录点数: {len(records)}")
    print(f"最后记录时间: {times[-1]:.3f} s")

    print("\n生成文件：")

    print(csv_path)
    print(trajectory_path)
    print(error_path)
    print(torque_path)

    print("==================================")


# ==========================================
# 9. 启动 MuJoCo Viewer
# ==========================================

mujoco.set_mjcb_control(controller)

print("\n==================================")
print("Day 7：任务空间 PD + 五次多项式轨迹跟踪")
print("==================================")

print("控制方式：")
print("F = Kp_task @ pos_error + Kd_task @ vel_error")
print("torque = J.T @ F")

print("\nRun：开始运动")
print("Pause：暂停运动")
print("Reset：重新从初始姿态开始")
print("关闭 Viewer：保存 CSV 和绘制图表")

print("==================================\n")

try:

    mujoco.viewer.launch(model, data)

finally:

    # 解除控制回调
    mujoco.set_mjcb_control(None)

    # 保存实验数据并绘图
    analyze_and_plot()


import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import csv
from pathlib import Path


# ==========================================
# 1. 参数
# ==========================================

INITIAL_Q1 = 0
INITIAL_Q2 =0

goal_pos = np.array([0.4, 0.8])

# 规划运动时间：4秒
T = 4.0

# 末端位置误差反馈增益
K_ik = 2.0

# 自适应阻尼参数
sigma_threshold = 0.1
lambda_max = 0.1

# 每个关节的期望速度限制
max_qdot = 1.0

# PD 参数
kp = np.array([60.0, 40.0])
kd = np.array([12.0, 8.0])

# 力矩限制
torque_min = np.array([-30.0, -20.0])
torque_max = np.array([30.0, 20.0])

# 输出文件保存到当前 Python 文件所在目录
OUTPUT_DIR = Path(__file__).resolve().parent


# ==========================================
# 2. MuJoCo XML
# ==========================================

XML = f"""
<mujoco model="adaptive_dls_tracking">

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
               ctrlrange="-30 30"/>

        <motor name="motor2"
               joint="joint2"
               ctrlrange="-20 20"/>

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

    if 0 <= t < T:

        tau = t / T

        s_dot = (
            30 * tau**2
            - 60 * tau**3
            + 30 * tau**4
        ) / T

        return s_dot * (goal_pos - start_pos)

    return np.zeros(2)


# ==========================================
# 5. 自适应阻尼函数
# ==========================================

def compute_adaptive_damping(J):

    # 计算雅可比矩阵的奇异值
    singular_values = np.linalg.svd(
        J,
        compute_uv=False
    )

    # 最小奇异值
    sigma_min = np.min(singular_values)

    # 根据奇异程度调整阻尼
    if sigma_min >= sigma_threshold:

        damping_now = 0.0

    else:

        damping_now = lambda_max * (
            1.0 - sigma_min / sigma_threshold
        )

    return damping_now, sigma_min


# ==========================================
# 6. 控制器状态
# ==========================================

# 积分得到的关节位置参考
q_ref = data.qpos[:2].copy()

# 上一次参考积分对应的仿真时间
last_ref_time = float(data.time)

# 上一次观察到的仿真时间
last_time = float(data.time)

# 上一次打印时间
last_print_time = -1.0


# ==========================================
# 7. 数据记录
# ==========================================

# 每条记录：
# [time, target_x, target_z,
#  actual_x, actual_z, error_mm,
#  sigma_min, damping]

records = []

last_record_time = -1.0


# ==========================================
# 8. 在线 IK + PD 控制器
# ==========================================

def controller(model, data):

    global q_ref
    global last_ref_time
    global last_time
    global last_print_time
    global last_record_time

    current_time = float(data.time)

    # --------------------------------------
    # A. 检测 Viewer Reset
    # --------------------------------------

    if current_time < last_time - 1e-9:

        q_ref = data.qpos[:2].copy()

        last_ref_time = current_time
        last_print_time = -1.0

        records.clear()
        last_record_time = -1.0

        print(
            "\n检测到 Reset："
            "控制器状态和实验数据已重置"
        )

    last_time = current_time

    # --------------------------------------
    # B. 读取当前状态
    # --------------------------------------

    q = data.qpos[:2].copy()

    q_dot = data.qvel[:2].copy()

    ee_pos = data.site_xpos[ee_id][[0, 2]].copy()

    # --------------------------------------
    # C. 计算期望轨迹
    # --------------------------------------

    target_pos = trajectory(current_time)

    error = target_pos - ee_pos

    error_mm = np.linalg.norm(error) * 1000.0

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
    # E. 在线逆运动学
    # --------------------------------------

    # 末端期望速度：
    # 轨迹前馈 + 位置误差反馈
    v_desired = (
        trajectory_velocity(current_time)
        + K_ik * error
    )

    # 根据当前雅可比矩阵计算自适应阻尼
    damping_now, sigma_min = (
        compute_adaptive_damping(J)
    )

    # 计算关节期望速度
    if damping_now == 0.0:

        # 远离奇异位形：使用伪逆
        q_dot_desired = (
            np.linalg.pinv(J) @ v_desired
        )

    else:

        # 接近奇异位形：使用 DLS
        A = (
            J @ J.T
            + damping_now**2 * np.eye(2)
        )

        q_dot_desired = (
            J.T @ np.linalg.solve(A, v_desired)
        )

    # 注意：无论走哪个分支，都要限幅
    q_dot_desired = np.clip(
        q_dot_desired,
        -max_qdot,
        max_qdot
    )

    # --------------------------------------
    # F. 根据真实仿真时间积分 q_ref
    # --------------------------------------

    elapsed = current_time - last_ref_time

    if elapsed > 0.0:

        q_ref += q_dot_desired * elapsed

    last_ref_time = current_time

    # --------------------------------------
    # G. PD 控制
    # --------------------------------------

    torque = (
        kp * (q_ref - q)
        + kd * (q_dot_desired - q_dot)
    )

    data.ctrl[:] = np.clip(
        torque,
        torque_min,
        torque_max
    )

    # --------------------------------------
    # H. 记录实验数据
    # --------------------------------------

    if current_time > last_record_time + 1e-9:

        records.append([
            current_time,
            float(target_pos[0]),
            float(target_pos[1]),
            float(ee_pos[0]),
            float(ee_pos[1]),
            float(error_mm),
            float(sigma_min),
            float(damping_now)
        ])

        last_record_time = current_time

    # --------------------------------------
    # I. 每 0.5 秒打印一次
    # --------------------------------------

    if current_time - last_print_time >= 0.5:

        last_print_time = current_time

        print(
            f"t={current_time:.2f}s | "
            f"target={np.round(target_pos, 4)} | "
            f"actual={np.round(ee_pos, 4)} | "
            f"error={error_mm:.2f} mm | "
            f"sigma_min={sigma_min:.4f} | "
            f"damping={damping_now:.4f} | "
            f"q_error={np.round(q_ref - q, 3)} | "
            f"torque={np.round(data.ctrl, 2)}"
        )


# ==========================================
# 9. 误差指标计算
# ==========================================

def calculate_metrics(errors):

    max_error = np.max(errors)

    mean_error = np.mean(errors)

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    return max_error, mean_error, rmse


# ==========================================
# 10. 保存数据并绘图
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

    sigma_values = data_array[:, 6]
    damping_values = data_array[:, 7]

    # --------------------------------------
    # A. 保存 CSV
    # --------------------------------------

    csv_path = OUTPUT_DIR / "tracking_data.csv"

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
            "sigma_min",
            "damping"
        ])

        writer.writerows(records)

    # --------------------------------------
    # B. 计算全程跟踪误差
    # --------------------------------------

    max_error, mean_error, rmse = (
        calculate_metrics(errors)
    )

    print("\n========== 全程跟踪误差 ==========")

    print(f"最大误差: {max_error:.3f} mm")
    print(f"平均误差: {mean_error:.3f} mm")
    print(f"RMSE:     {rmse:.3f} mm")

    # --------------------------------------
    # C. 计算 0～4 秒运动阶段误差
    # --------------------------------------

    motion_mask = times <= T

    motion_errors = errors[motion_mask]

    if len(motion_errors) > 0:

        motion_max, motion_mean, motion_rmse = (
            calculate_metrics(motion_errors)
        )

        print("\n========== 运动阶段误差 ==========")

        print(f"统计区间: 0 ~ {T:.1f} s")

        print(f"最大误差: {motion_max:.3f} mm")
        print(f"平均误差: {motion_mean:.3f} mm")
        print(f"RMSE:     {motion_rmse:.3f} mm")

    # --------------------------------------
    # D. 绘制轨迹对比图
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
        "End-Effector Trajectory Tracking"
    )

    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    trajectory_path = (
        OUTPUT_DIR / "trajectory_tracking.png"
    )

    plt.savefig(
        trajectory_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------
    # E. 绘制位置跟踪误差曲线
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
        "End-Effector Tracking Error"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    error_path = (
        OUTPUT_DIR / "tracking_error.png"
    )

    plt.savefig(
        error_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------
    # F. 绘制自适应阻尼变化曲线
    # --------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(9, 7),
        sharex=True
    )

    axes[0].plot(
        times,
        sigma_values,
        label="Minimum singular value"
    )

    axes[0].axhline(
        y=sigma_threshold,
        color="red",
        linestyle="--",
        label="Threshold"
    )

    axes[0].set_ylabel("Sigma min")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        times,
        damping_values,
        label="Adaptive damping"
    )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Damping")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(
        "Adaptive Damping During Tracking"
    )

    plt.tight_layout()

    damping_path = (
        OUTPUT_DIR / "adaptive_damping.png"
    )

    plt.savefig(
        damping_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------
    # G. 打印保存结果
    # --------------------------------------

    print("\n========== 自适应阻尼统计 ==========")

    print(
        f"最小奇异值范围: "
        f"{np.min(sigma_values):.5f} ~ "
        f"{np.max(sigma_values):.5f}"
    )

    print(
        f"阻尼范围: "
        f"{np.min(damping_values):.5f} ~ "
        f"{np.max(damping_values):.5f}"
    )

    print("\n========== 数据保存完成 ==========")

    print(f"记录点数: {len(records)}")
    print(f"最后记录时间: {times[-1]:.3f} s")

    print("\n生成文件：")

    print(csv_path)
    print(trajectory_path)
    print(error_path)
    print(damping_path)

    print("==================================")


# ==========================================
# 11. 启动 MuJoCo Viewer
# ==========================================

mujoco.set_mjcb_control(controller)

print("\n==================================")
print("Day6：自适应阻尼 + 五次多项式轨迹跟踪")
print("==================================")

print("Run：开始运动")
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

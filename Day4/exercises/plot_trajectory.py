
import numpy as np
import matplotlib.pyplot as plt

# ======================================
# 1. 参数
# ======================================

T = 4.0

t = np.linspace(0, T, 1001)

tau = t / T

# ======================================
# 2. 线性插值
# ======================================

s_linear = tau

v_linear = np.ones_like(t) / T

a_linear = np.zeros_like(t)

# ======================================
# 3. 五次多项式插值
# ======================================

s_quintic = (
    10 * tau**3
    - 15 * tau**4
    + 6 * tau**5
)

v_quintic = (
    30 * tau**2
    - 60 * tau**3
    + 30 * tau**4
) / T

a_quintic = (
    60 * tau
    - 180 * tau**2
    + 120 * tau**3
) / T**2

# ======================================
# 4. 绘制进度曲线
# ======================================

plt.figure(figsize=(8, 5))

plt.plot(
    t, s_linear,
    label="Linear",
    linestyle="--"
)

plt.plot(
    t, s_quintic,
    label="Quintic"
)

plt.xlabel("Time (s)")
plt.ylabel("Path progress")

plt.title("Trajectory Progress Comparison")

plt.legend()
plt.grid(True)

plt.savefig(
    "day5_position.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ======================================
# 5. 绘制速度曲线
# ======================================

plt.figure(figsize=(8, 5))

plt.plot(
    t, v_linear,
    label="Linear",
    linestyle="--"
)

plt.plot(
    t, v_quintic,
    label="Quintic"
)

plt.xlabel("Time (s)")
plt.ylabel("Progress rate (1/s)")

plt.title("Velocity Profile Comparison")

plt.legend()
plt.grid(True)

plt.savefig(
    "day5_velocity.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ======================================
# 6. 绘制加速度曲线
# ======================================

plt.figure(figsize=(8, 5))

plt.plot(
    t, a_linear,
    label="Linear",
    linestyle="--"
)

plt.plot(
    t, a_quintic,
    label="Quintic"
)

plt.xlabel("Time (s)")
plt.ylabel("Progress acceleration (1/s^2)")

plt.title("Acceleration Profile Comparison")

plt.legend()
plt.grid(True)

plt.savefig(
    "day5_acceleration.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ======================================
# 7. 输出关键数据
# ======================================

print("===== Day5 轨迹规划对比 =====")

print(f"总运动时间: {T:.1f} s")

print(
    "线性插值最大进度速率:",
    np.max(v_linear)
)

print(
    "五次多项式最大进度速率:",
    np.max(v_quintic)
)

print(
    "五次多项式初始速度:",
    v_quintic[0]
)

print(
    "五次多项式末端速度:",
    v_quintic[-1]
)

print("\n图片已保存：")
print("day5_position.png")
print("day5_velocity.png")
print("day5_acceleration.png")

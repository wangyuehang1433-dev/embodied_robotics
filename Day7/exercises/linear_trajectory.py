
import numpy as np

# 1. 起点和终点（XZ平面）
start = np.array([0.2, 1.0])
goal = np.array([0.4, 0.8])

# 2. 轨迹总时间：2秒
T = 2.0

# 3. 每0.5秒计算一次位置
times = np.linspace(0, T, 5)

# 4. 生成直线轨迹
for t in times:
    s = t / T

    p_desired = (1 - s) * start + s * goal

    print(
        f"t={t:.1f}s, "
        f"s={s:.2f}, "
        f"p={p_desired}"
    )

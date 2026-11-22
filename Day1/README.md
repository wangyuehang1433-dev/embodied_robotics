# Day 1｜MuJoCo 环境、平面双关节机械臂与正运动学

日期：2026-11-18  
阶段：半年具身智能实习准备 · 机械臂控制入门  
路线：机械臂控制 → 强化学习 → 模仿学习 → 真机项目 → 运动控制拓展

## 环境

Windows 11，Conda 环境 `embodied_day1`，Python 3.12，MuJoCo 4.0.1，已安装 NumPy 和 Matplotlib，使用 VS Code。机器没有 NVIDIA GPU，本次物理仿真在 CPU 上运行。已成功导入依赖。

VS Code 的 PowerShell 起初无法识别 `conda`。目前可以选中 Conda 环境的 Python 解释器，也可以直接用其绝对路径运行脚本。要找出实际路径，在能够激活环境的终端运行：

```powershell
conda activate embodied_day1
python -c "import sys; print(sys.executable)"
```

在 VS Code 使用「Python: Select Interpreter」选择该路径，或在项目根目录运行：

```powershell
& '<上一步输出的 python.exe 绝对路径>' .\Day1\forward_kinematics.py
```

## 实验与验收

### 1. 小球自由落体

- **目的**：确认 MuJoCo 能加载模型、推进仿真并处理碰撞。
- **原理**：重力使小球在接触前向下加速。每次 `mj_step` 推进一个时间步，小球接触地面后由接触力改变运动。
- **操作**：加载小球和地面模型，创建 `MjData`，循环调用 `mj_step`，读取 `data.time` 和小球自由关节的竖直位置。
- **验收**：已运行。实际仿真时间为 `2.0000000000000013` s，实际小球高度为 `0.09963281815747682` m。时间末位的差异来自浮点表示。当前项目目录未找到独立的自由落体脚本。

### 2. 双关节机械臂与 PD 控制

- **目的**：建立关节、连杆、执行器与末端之间的关系，观察位置反馈驱动关节运动。
- **原理**：两段连杆长 0.5 m、0.4 m，基座高 0.2 m。两个 Hinge 关节绕各自局部 Y 轴旋转，带动连杆在 X–Z 平面运动。两个 Motor 提供输入，模型重力为零，时间步长为 0.002 s。
- **操作**：在 [two_link_arm.py](two_link_arm.py) 中用 MJCF 建模，读取 `qpos`、`qvel`，按 PD 公式计算力矩并限幅，写入 `ctrl`，执行 `mj_step` 并同步 Viewer。Day 1 学习记录中的目标为 `[0.6, -0.8]` rad。
- **验收**：机械臂仿真已运行。未记录最终角度和跟踪误差。当前文件中的初始关节角为 `[0.6, 0.8]`，PD 目标为 `[0.6, 0]`，因此需要先统一参数再复现实验。内嵌 MJCF 的 `</actuator>` 后有一个 `·`，也列入复跑检查。

### 3. Viewer 与状态观察

- **目的**：学会从界面定位仿真状态、关节和执行器。
- **操作**：浏览 File、Option、Simulation、Joint、Control、Watch、Physics、Physical Parameters、Rendering、Visualization 和 Logging。用 Joint、Control 和 Watch 建立界面读数与 `qpos`、`qvel`、`ctrl` 的对应关系。
- **验收**：已熟悉面板及 Site、Joint、Geom、Actuator 的用途。当前目录保存了 [仿真截图](results/screenshot.png)。

### 4. 平面正运动学（Forward Kinematics, FK）

- **目的**：给定两关节角，求末端在世界坐标系中的 X、Z 坐标。
- **操作**：运行 [forward_kinematics.py](forward_kinematics.py)，输入 `q1 = 0.6`、`q2 = -0.8` rad。将两段连杆的平面投影和基座高度相加。
- **验收**：已运行。实际输出 X = `0.20285350437949318` m，Z = `1.004694438591336` m。与 MuJoCo 的 `end_effector` Site 在相同角度下的逐项对照留待 Day 2。

### 5. 雅可比与有限差分

- **目的**：把关节速度映射到末端线速度，并验证手推偏导。
- **操作**：[jacobian.py](jacobian.py) 含解析矩阵和前向有限差分。计划在 `q = [0.6, -0.8]` rad、`q_dot = [0.5, 0.2]` rad/s 下运行，比较两种矩阵并打印末端速度。
- **验收**：待验证。已学习偏导、有限差分、伪逆和奇异性的概念。没有确认脚本输出，也没有伪逆实验结果。

### 6. 坐标系与齐次变换

- **目的**：理解同一点在不同坐标系中的表示，以及变换链如何把末端局部坐标换算到世界坐标。
- **操作记录**：接触了世界、基座、关节和末端坐标系，以及旋转矩阵、齐次变换。按学习记录，`coordinate_transform.py` 示例已编写，但当前提交目录中未找到该文件，未确认运行结果。
- **验收**：**未掌握。** Day 2 从二维手算开始，随后用 NumPy 验证点变换、变换复合和逆变换。

## 数学公式与推导

**时间步进**。固定步长下 $t_N=N\Delta t$。理想自由落体在接触前、初速度为零时满足 $z(t)=z(0)-gt^2/2$。接触后以实际仿真状态为准。

**关节 PD**。令 $e=q_d-q$。固定目标时 $\dot e=-\dot q$，代入 $\tau=K_p e+K_d\dot e$ 得到 $\tau=K_p(q_d-q)-K_d\dot q$。当前代码按关节逐项计算，力矩限幅分别为 ±30、±20。比例项随位置误差增大，微分项抵抗当前速度。

**FK**。零角时连杆沿 +Z，绕 +Y 转动 $\theta$ 后，长为 $L$ 的向量在 X、Z 方向上的投影分别为 $L\sin\theta$、$L\cos\theta$。第二个关节角相对第一段连杆，因此第二段的绝对角为 $q_1+q_2$。从基座到末端相加：

$$
x=L_1\sin q_1+L_2\sin(q_1+q_2),\qquad
z=z_0+L_1\cos q_1+L_2\cos(q_1+q_2).
$$

**位置雅可比**。对 $p=[x,z]^T$ 使用链式法则，得到 $\dot p=J(q)\dot q$，第 $i$ 列是对 $q_i$ 的偏导。对 $q_1$ 求导时两段投影都变化，对 $q_2$ 求导时只有第二段变化：

$$
J=
\begin{bmatrix}
L_1\cos q_1+L_2\cos(q_1+q_2)&L_2\cos(q_1+q_2)\\
-L_1\sin q_1-L_2\sin(q_1+q_2)&-L_2\sin(q_1+q_2)
\end{bmatrix}.
$$

脚本用 $[p(q+\epsilon e_i)-p(q)]/\epsilon$ 近似第 $i$ 列，$\epsilon=10^{-6}$ rad。将两列行列式展开，第二段自身的交叉项抵消，剩余 $\det J=-L_1L_2\sin q_2$。两连杆伸直或折叠时秩下降。理论上可用 $\dot q=J^+\dot p_d$ 求最小范数的最小二乘解，接近奇异位形时还需要观察关节速度是否放大。

**齐次变换（待重学）**。采用列向量约定，$T_{WB}$ 将基座坐标变换到世界坐标，$T_{BE}$ 将末端坐标变换到基座坐标。对一个末端局部点，先算 $\bar p_B=T_{BE}\bar p_E$，再算 $\bar p_W=T_{WB}\bar p_B$，因此 $T_{WE}=T_{WB}T_{BE}$。将矩阵分成旋转 $R$ 和位移 $t$ 后可写为 $p_W=R_{WB}p_B+t_{WB}$。由此解出 $p_B=R_{WB}^T(p_W-t_{WB})$，逆变换的平移块为 $-R_{WB}^T t_{WB}$。这是 Day 2 的推导参考，尚未通过本人实践验收。

## Python 与 MuJoCo API

| API 或元素 | 本项目中的含义 |
| --- | --- |
| `mujoco.MjModel.from_xml_string`、`mujoco.MjData` | 从 MJCF 构造模型，为运行中的状态分配数据 |
| `mujoco.mj_step`、`data.time`、`model.opt.timestep` | 执行单步，读取仿真时间和步长 |
| `data.qpos`、`data.qvel`、`data.ctrl` | 广义位置、广义速度、执行器控制输入。双 Hinge 模型的前两项是一对关节角和角速度。自由关节的姿态采用四元数，位置向量和速度向量长度可不同 |
| `mujoco.mj_forward` | 手动写入关节角后刷新派生状态，无需推进时间 |
| `mj_name2id`、`mjtObj.mjOBJ_SITE`、`data.site_xpos` | 按名称找末端 Site，读取它的世界坐标，Day 2 先检查 ID 有效 |
| `mujoco.viewer.launch_passive`、`viewer.is_running`、`viewer.sync` | 打开 Viewer 并把脚本中的状态同步到窗口 |
| MJCF `body`、`joint`、`geom`、`site`、`motor` | 刚体层级、运动自由度、形状与碰撞几何、附着标记、执行器 |
| `np.array`、`np.sin`、`np.cos`、`np.clip` | 构造向量、计算投影、限制力矩 |
| `@ `、`np.zeros`、`np.max`、`np.abs` | 矩阵乘法、创建数值雅可比、统计最大绝对误差 |
| `np.linalg.pinv`、`np.linalg.svd`、Matplotlib | Day 2 用于伪逆、奇异值检查和跟踪曲线 |

## 任务清单

| 状态 | 内容 |
| --- | --- |
| 已完成 | 环境与库导入、绝对路径运行、小球仿真、双关节臂建模和仿真、Viewer 基础、独立 FK 计算 |
| 待验证 | 用统一参数复跑 PD、FK 与 MuJoCo Site 对照、Jacobian 有限差分与伪逆实验 |
| 未掌握 | 坐标系方向、局部点换算、齐次变换复合与求逆。示例脚本实际输出也待确认 |

## 项目目录建议

保留现有 `embodied_robotics/Day1/`。日志和 Day 1 的示例继续放在该目录，方便从记录跳转到实际代码。

```text
embodied_robotics/
├── Day1/
│   ├── README.md
│   ├── two_link_arm.py
│   ├── forward_kinematics.py
│   ├── jacobian.py
│   ├── models/mjmodel.xml
│   ├── exports/MJMODEL.TXT
│   ├── results/screenshot.png
│   └── .vscode/settings.json
└── Day2/                         # 计划新增
    ├── coordinate_transform.py    # 找回示例后补齐或重写
    ├── verify_kinematics.py        # FK、Jacobian 和 Site 交叉验证
    └── joint_tracking.png         # 运行之后生成
```

## Day 2 计划（2026-11-19，约 3 小时）

| 时间 | 工作 | 验收标准 |
| --- | --- | --- |
| 20 分钟 | 沿用 `embodied_day1`。检查内嵌 MJCF，明确记录初始姿态与 PD 目标，复跑 Viewer | 输出 Python 和 MuJoCo 版本，模型加载成功，关节与执行器数量均为 2，窗口可正常关闭 |
| 50 分钟 | 重学坐标系。先画世界、基座和末端三组轴，手算局部原点和单位轴，再用 NumPy 实现变换链 | 能用自己的话说明 $T_{WB}$ 的方向。只平移示例结果与手算一致，绕 +Y 旋转 90° 时 +Z 映射到 +X。两步变换与复合矩阵结果一致，往返误差小于 $10^{-9}$ m |
| 30 分钟 | 用相同关节角比较解析 FK 与 MuJoCo Site。先测零位，再测 `[0.6, -0.8]`，每次写入 `qpos` 后执行 `mj_forward` | 零位理论位置 X 为 0、Z 为 1.1 m。两组姿态的 Site 与解析 FK 在 X、Z 上的最大绝对差小于 $10^{-6}$ m |
| 40 分钟 | 运行 `jacobian.py`，检查有限差分。对比一般姿态和伸直位形，观察伪逆和最小奇异值 | 一般姿态解析与数值 Jacobian 最大绝对误差小于 $10^{-5}$ m/rad。能用奇异值解释伸直时缺失的速度方向，并用 $\lVert J\dot q-\dot p_d\rVert$ 报告伪逆残差 |
| 40 分钟 | 记录约 5 秒的 PD 跟踪，保存关节角曲线和目标线 | PNG 含时间、角度单位和两条目标线。打印最终角、目标角和每关节误差。以两关节最终绝对误差均小于 0.02 rad 为验收线，失败时保留输出并记录排查结论 |

## 实习技能积累

Day 1 的可展示成果是一个在 Windows 上运行的双自由度机械臂仿真，以及一份有实测输出的 FK 脚本。机械臂控制阶段用 PD 建立反馈基线，用 Site 和关节状态校验模型。进入强化学习后，可沿用这些量来定义 observation、action 和 reward。模仿学习会用到带时间戳的关节、末端和动作轨迹。真机和运动控制拓展还依赖坐标系标定、运动约束以及奇异位形的判断，这些能力应通过后续实验补齐。

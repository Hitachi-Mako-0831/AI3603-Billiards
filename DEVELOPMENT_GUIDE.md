# AI3603 Billiards RL 项目开发指南

本指南旨在帮助小组成员快速上手项目架构，并行开发不同的 RL 算法。

## 1. 快速开始 (Quick Start)

### 1.1 环境安装

本项目依赖 `pooltool` 和 `gymnasium`。由于 `pooltool` 依赖特定的 `panda3d` 版本，请严格按照以下命令安装：

```bash
# 1. 安装 Panda3D (指定版本)
pip install --extra-index-url https://archive.panda3d.org/ panda3d==1.11.0.dev3702

# 2. 安装其他依赖
pip install pooltool-billiards==0.5.0 gymnasium bayesian-optimization scikit-learn
```

### 1.2 运行测试

使用随机策略验证环境是否配置正确：

```bash
python train_manager.py --algo random
```

如果看到 "Sanity check passed!" 字样，说明环境一切正常。

---

## 2. 项目架构 (Architecture)

我们采用了 **Registry Pattern** (注册工厂模式)，这意味着你可以专注于写自己的算法文件，而不需要修改主程序的代码。

```text
AI3603-Billiards/
├── algorithms/             # [算法仓库] 所有的 Agent 都在这里
│   ├── base.py             # 所有 Agent 必须继承的基类
│   ├── registry.py         # 注册器
│   ├── random_agent.py     # 示例
│   └── ppo_agent/          # (示例) 你可以在这里新建文件夹放你的算法
│
├── envs/                   # [环境封装]
│   └── wrapper.py          # 将 PoolEnv 转换为 Gym 接口 (State -> Tensor)
│
├── configs/                # [配置文件] 存放 yaml 配置
├── train_manager.py        # [统一入口] 负责加载配置并运行
└── poolenv.py              # [原始环境] 不可修改
```

---

## 3. 如何添加新算法？ (How to Add New Algorithm)

假设你要添加一个 **PPO** 算法，请遵循以下步骤：

### Step 1: 创建文件

在 `algorithms/` 目录下创建一个新文件（或文件夹），例如 `algorithms/ppo_agent.py`。

### Step 2: 继承并注册

你的 Agent 类必须继承 `BasePolicy`，并使用 `@register_algorithm` 装饰器。

```python
# algorithms/ppo_agent.py

from .base import BasePolicy
from .registry import register_algorithm
import numpy as np

@register_algorithm("ppo")  # <--- 给你的算法起个名字
class PPOAgent(BasePolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 在这里初始化你的神经网络
        # self.actor = ...
        # self.critic = ...
        print(f"PPO Agent initialized with config: {cfg}")

    def select_action(self, observation, evaluate=False):
        # observation 是一个 (79,) 的 numpy 数组
        # 你的输出应该是一个 (5,) 的 numpy 数组，范围 [-1, 1]
        
        # 示例: 随机动作
        action = np.random.uniform(-1, 1, size=(5,))
        return action
        
    def train(self, replay_buffer):
        # 在这里写训练逻辑
        pass
```

### Step 3: 导出模块

在 `algorithms/__init__.py` 中添加一行导入，以便注册器能扫描到你的文件：

```python
# algorithms/__init__.py
from .registry import register_algorithm, get_algorithm_class
from .base import BasePolicy
from .random_agent import RandomAgent

from .ppo_agent import PPOAgent  # <--- 添加这一行
```

### Step 4: 运行

现在你可以直接在命令行调用你的算法了：

```bash
python train_manager.py --algo ppo
```

还可以传入配置文件：

```bash
python train_manager.py --algo ppo --config configs/ppo_config.yaml
```

---

## 4. 接口说明 (API Reference)

### 4.1 观测空间 (Observation Space)

`envs/wrapper.py` 会将环境状态转换为一个 79 维的向量：

- `[0:4]`: 白球 (x, y, vx, vy)
- `[4:64]`: 1-15号球 (x, y, vx, vy) * 15
- `[64:79]`: 目标球指示器 (1.0 表示该球是我的目标，0.0 表示不是)

### 4.2 动作空间 (Action Space)

动作是一个 5 维向量，取值范围均为 `[-1, 1]`。Wrapper 会自动将其映射到物理参数：

| 维度 | 含义 | 映射范围 | 说明 |
| :--- | :--- | :--- | :--- |
| 0 | V0 (力度) | [0.5, 8.0] m/s | 击球速度 |
| 1 | phi (水平角) | [0, 360] 度 | 瞄准方向 |
| 2 | theta (垂直角) | [0, 90] 度 | 扎杆角度 (通常很小) |
| 3 | a (水平击球点) | [-0.5, 0.5] | 塞 (English) |
| 4 | b (垂直击球点) | [-0.5, 0.5] | 高低杆 |

### 4.3 奖励函数 (Reward Function) - **关键**

Wrapper 的核心作用不仅仅是接口转换，更重要的是定义 **Reward**。目前的实现非常基础：

- 进一个目标球: `+1.0`
- 进一个对手球: `-0.5`
- 白球进袋: `-1.0`
- 赢得比赛: `+10.0`
- 输掉比赛: `-10.0`

**改进建议**:
为了训练出更强的 Agent，强烈建议修改 `envs/wrapper.py` 中的 `step` 函数，加入 **Dense Reward** (稠密奖励)，例如：
- 击球后母球是否停到了更有利的位置？
- 目标球是否被撞击并靠近了袋口？
- 是否解到了斯诺克？

### 4.4 训练难度与速度调整 (Training Difficulty & Speed)

为了辅助 Agent 学习以及加快训练速度，我们可以动态调整对手 (BasicAgent) 的能力和计算量。

**1. 调整对手噪声 (难度控制)**
噪声越大，对手越弱（失误越多）。

- `--enable_opponent_noise`: 开启对手噪声
- `--opponent_noise_scale <float>`: 噪声放大倍率（默认 1.0）

**2. 调整对手思考速度 (速度控制)**
默认情况下，BasicAgent 会进行 30 次物理模拟来寻找最佳击球，这非常耗时。开启快速模式可以将模拟次数降至 7 次。

- `--fast_opponent`: 开启快速对手模式（搜索次数从 20+10 降为 5+2），显著提高训练速度，但对手能力会变弱。
- `--max_timesteps <int>`: 指定训练总步数（覆盖配置文件），例如 `--max_timesteps 5000`。

**示例**:
```bash
# 1. 默认训练 (对手最强，速度最慢)
python train_manager.py --mode train --algo ppo

# 2. 快速训练 (对手较弱，速度快，适合调试或初期训练)
python train_manager.py --mode train --algo ppo --fast_opponent --max_timesteps 10000

# 3. 开启高噪声 (对手很菜)
python train_manager.py --mode train --algo ppo --enable_opponent_noise --opponent_noise_scale 5.0

### 4.5 断点续训 (Resume Training)

如果训练被中断，或者想在已有模型的基础上继续训练，可以使用 `--load_model` 参数。

```bash
# 从指定的 checkpoint 继续训练
python train_manager.py --mode train --algo ppo --load_model checkpoints/ppo_interrupted.pth
```

### 4.6 日志与可视化 (Logging & Visualization)

训练过程会自动记录到 `logs/` 目录下的 CSV 文件中（例如 `logs/ppo_training_log.csv`）。
记录的字段包括：Episode, Step, Reward, ActorLoss, CriticLoss, TotalLoss。

我们提供了一个脚本 `plot_training.py` 来快速绘制训练曲线：

```bash
# 绘制 PPO 训练日志，默认输出到 logs/ppo_training_log_plot.png
python plot_training.py --file logs/ppo_training_log.csv

# 指定输出文件和抗锯齿平滑窗口大小
python plot_training.py --file logs/ppo_training_log.csv --output results/my_plot.png --smooth 50
```

生成的图片包含两个子图：
1. **Reward**: 包含原始奖励（透明）和移动平均奖励（实线）。
2. **Loss**: 包含 Actor Loss, Critic Loss 和 Total Loss 的变化曲线。

---

## 5. 小组分工建议

*   **成员 A**: 负责完善 `envs/wrapper.py` 中的 Reward Function，尝试设计更合理的奖励机制。
*   **成员 B**: 尝试实现一个 Off-policy 算法 (如 TD3 或 SAC)。
*   **成员 C**: 尝试实现一个 Evolution Strategy (如 CMA-ES)，这在物理模拟中通常很有效。

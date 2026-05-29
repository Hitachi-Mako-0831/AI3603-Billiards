# AI3603-Billiards

AI3603-Billiards 是一个基于 `pooltool` 的八球台球对战项目，提供了完整的台球规则环境、基准智能体、混合智能体，以及一个用于自我对弈训练价值网络的实验框架。项目的目标是让智能体在带噪声的物理环境中完成稳定、可复现的击球决策，并在评测对战中获得尽可能高的得分。

## 项目内容

- `poolenv.py`：台球环境与规则判定，负责重置局面、执行击球、交换球权和终局判断。
- `evaluate.py`：评测脚本，用于让两个智能体进行多局对战并统计胜负和得分。
- `agents/`：智能体实现目录，包含基准智能体、进阶智能体和当前提交的 `NewAgent`。
- `train/rl_trainer.py`：自我对弈 + TD 学习训练框架，用于训练局面价值网络。
- `GAME_RULES.md`：台球规则与评测细则说明。
- `PROJECT_GUIDE.md`：课程任务说明与提交要求。
- `requirements.txt`：依赖列表。

## 环境要求

- 操作系统：推荐 Ubuntu 22.04
- Python：建议 3.10 及以上。
- 依赖：`pooltool-billiards`、`numpy`、`bayesian-optimization`、`torch`、`scikit-learn`。

> 注意：项目中使用了 `signal.SIGALRM` 做物理模拟超时保护，原生 Windows 可能不兼容。若你当前在 Windows 上开发，推荐通过 WSL2 运行，或者在 Linux 环境中执行训练和评测。

## 安装

在项目根目录下创建并激活虚拟环境后安装依赖：

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux / WSL
source .venv/bin/activate

pip install -r requirements.txt
```

## 目录结构

```text
AI3603-Billiards/
├── evaluate.py
├── GAME_RULES.md
├── poolenv.py
├── PROJECT_GUIDE.md
├── README.md
├── requirements.txt
├── utils.py
├── agents/
│   ├── __init__.py
│   ├── agent.py
│   ├── basic_agent.py
│   ├── basic_agent_pro.py
│   └── new_agent.py
└── train/
	└── rl_trainer.py
```

## 快速开始

### 1. 运行评测

默认情况下，`evaluate.py` 会让 `BasicAgent` 和 `NewAgent` 进行对战，并按照 4 局为一个循环轮换先后手与球型，确保公平性。

```bash
python evaluate.py
```

评测结果会输出类似如下的统计信息：

```text
最终结果： {'AGENT_A_WIN': 15, 'AGENT_B_WIN': 23, 'SAME': 2, 'AGENT_A_SCORE': 16.0, 'AGENT_B_SCORE': 24.0}
```

其中得分计算方式为：

- 胜一局记 1 分
- 平局双方各记 0.5 分
- 总分越高，说明该智能体的综合表现越好

### 2. 切换对手

如果你想测试不同基线，只需要修改 `evaluate.py` 中的对战对象：

- `BasicAgent()`：课程提供的基准智能体
- `BasicAgentPro()`：基于 MCTS 的进阶智能体
- `NewAgent()`：当前提交的智能体

### 3. 运行训练

训练脚本位于 `train/rl_trainer.py`，默认使用自我对弈收集数据，并通过价值网络做 TD 学习。建议使用模块方式启动，确保可以正确导入仓库中的包：

```bash
python -m train.rl_trainer --episodes 500 --batch-size 64 --lr 1e-4 --save-dir checkpoints
```

常用参数：

- `--episodes`：训练局数，默认 `500`
- `--batch-size`：训练批大小，默认 `64`
- `--lr`：学习率，默认 `1e-4`
- `--save-dir`：检查点保存目录，默认 `checkpoints`
- `--resume`：从指定检查点继续训练

训练过程中会保存：

- `value_net_ep*.pt`：阶段性价值网络检查点
- `replay_buffer_ep*.npy`：经验回放缓冲区
- `value_net_final.pt`：最终模型

### 4. 在智能体中加载训练好的价值网络

如果你已经训练好了价值网络，可以在 `agents/new_agent.py` 中启用学习型评估器：

```python
from agents import NewAgent

agent = NewAgent(
	use_value_network=True,
	value_net_path="checkpoints/value_net_final.pt"
)
```

## 规则与评测说明

- 游戏使用标准八球规则，共 16 个球。
- 母球不能进袋，黑 8 球必须在己方目标球清空后才可合法打进。
- `evaluate.py` 默认执行 40 局对战，并通过 `AGENT_A_SCORE` / `AGENT_B_SCORE` 统计最终分数。
- 环境中的噪声来自 `poolenv.py`，用于模拟真实击球误差，正式评测时不应修改环境规则。

## 智能体简介

- `BasicAgent`：基于贝叶斯优化和物理模拟的基准智能体。
- `BasicAgentPro`：基于 MCTS 的进阶智能体。
- `NewAgent`：当前提交的混合智能体，结合启发式候选生成、快速仿真筛选和可选的学习型局面评估器。

## 常见问题

### 为什么建议在 Linux / WSL2 下运行？

因为项目里使用了 `signal.SIGALRM` 做模拟超时控制，这个机制在原生 Windows 上通常不可用。

### 训练时为什么要使用模块方式启动？

因为训练脚本需要从仓库根目录导入 `agents`、`poolenv` 等模块，使用 `python -m train.rl_trainer` 更稳妥。

### 如何查看更详细的规则？

请参考 `GAME_RULES.md`，其中包含球型分配、犯规判定和计分规则的完整说明。


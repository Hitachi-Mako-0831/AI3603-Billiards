from .agent import Agent
from .basic_agent import BasicAgent
from .basic_agent_pro import BasicAgentPro
try:
    from .new_agent import NewAgent
except ModuleNotFoundError as e:
    if getattr(e, "name", None) != "torch":
        raise

    class NewAgent(Agent):
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("NewAgent 需要安装 torch 才能使用")

        def decision(self, balls, my_targets, table):
            raise ModuleNotFoundError("NewAgent 需要安装 torch 才能使用")

try:
    from .ppo_agent import PPOAgent
except ModuleNotFoundError as e:
    if getattr(e, "name", None) != "torch":
        raise

    class PPOAgent(Agent):
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("PPOAgent 需要安装 torch 才能使用")

        def decision(self, balls, my_targets, table):
            raise ModuleNotFoundError("PPOAgent 需要安装 torch 才能使用")

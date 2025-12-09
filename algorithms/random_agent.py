"""
Random Agent - 用于测试架构连通性
"""

import numpy as np
from .base import BasePolicy
from .registry import register_algorithm

@register_algorithm("random")
class RandomAgent(BasePolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        print("[RandomAgent] Initialized with config:", cfg)
        
    def select_action(self, observation, evaluate=False):
        # 模拟一个随机动作
        # V0: [0.5, 8.0], phi: [0, 360], theta: [0, 90], a: [-0.5, 0.5], b: [-0.5, 0.5]
        action = np.array([
            np.random.uniform(0.5, 8.0),
            np.random.uniform(0, 360),
            np.random.uniform(0, 90),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5)
        ])
        return action

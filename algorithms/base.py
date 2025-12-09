"""
算法基类
"""

class BasePolicy:
    def __init__(self, cfg):
        self.cfg = cfg

    def select_action(self, observation, evaluate=False):
        """
        根据观测选择动作
        
        参数:
            observation: np.ndarray (Gym Wrapper 处理后的向量)
            evaluate: bool, 是否为评估模式 (True则不使用随机探索)
            
        返回:
            action: np.ndarray or dict (取决于环境接口)
        """
        raise NotImplementedError
    
    def train(self, replay_buffer, batch_size=256):
        """
        训练接口 (可选)
        """
        pass

    def save(self, path):
        """保存模型"""
        pass

    def load(self, path):
        """加载模型"""
        pass

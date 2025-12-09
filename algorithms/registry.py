"""
算法注册工厂
"""

_ALGORITHM_REGISTRY = {}

def register_algorithm(name):
    """
    装饰器：注册一个算法类
    
    @register_algorithm("ppo")
    class PPOAgent(BaseAgent):
        ...
    """
    def decorator(cls):
        if name in _ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm '{name}' is already registered!")
        _ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator

def get_algorithm_class(name):
    """根据名称获取算法类"""
    if name not in _ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' not found. Available: {list(_ALGORITHM_REGISTRY.keys())}")
    return _ALGORITHM_REGISTRY[name]

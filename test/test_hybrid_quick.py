"""
快速测试 HybridAgent 单次决策
"""

from poolenv import PoolEnv
from agent import HybridAgent
import time

def main():
    print("=" * 60)
    print("HybridAgent 快速测试")
    print("=" * 60)
    
    # 创建环境和Agent
    env = PoolEnv(verbose=True, record_shots=False)
    agent = HybridAgent(use_value_network=False)
    
    # 重置环境
    env.reset(target_ball='solid')
    
    # 获取观测
    balls, my_targets, table = env.get_observation(player='A')
    
    print(f"\n目标球: {my_targets}")
    print(f"白球位置: {balls['cue'].state.rvw[0][:2]}")
    print(f"目标球数量: {len([b for b in my_targets if balls[b].state.s != 4])}")
    
    # 测试决策
    print("\n开始决策...")
    start_time = time.time()
    
    action = agent.decision(balls=balls, my_targets=my_targets, table=table)
    
    elapsed = time.time() - start_time
    
    print(f"\n决策完成 (耗时: {elapsed:.2f}秒)")
    print(f"动作: V0={action['V0']:.2f}, phi={action['phi']:.1f}°, " 
          f"theta={action['theta']:.1f}°, a={action['a']:.3f}, b={action['b']:.3f}")
    
    # 执行动作
    print("\n执行击球...")
    env.take_shot(action)
    
    # 检查结果
    done, info = env.get_done()
    print(f"\n游戏是否结束: {done}")
    if done:
        print(f"结果信息: {info}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

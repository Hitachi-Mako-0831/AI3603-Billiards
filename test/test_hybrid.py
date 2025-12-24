"""
测试 HybridAgent vs BasicAgent
"""

from poolenv import PoolEnv
from agent import BasicAgent, HybridAgent

def main():
    """运行对战测试"""
    print("=" * 60)
    print("HybridAgent vs BasicAgent 对战测试")
    print("=" * 60)
    
    # 创建环境
    env = PoolEnv(verbose=True, record_shots=False)
    
    # 创建 Agent
    agent_a = HybridAgent(use_value_network=False)  # 不使用学习型评估器
    agent_b = BasicAgent()
    
    agents = [agent_a, agent_b]
    agent_names = ["HybridAgent", "BasicAgent"]
    
    # 运行多局测试
    num_games = 5
    wins = [0, 0]
    
    for game_idx in range(num_games):
        print(f"\n{'='*60}")
        print(f"第 {game_idx + 1} 局")
        print(f"{'='*60}")
        
        # 重置环境 - 轮换球型和先后手
        target_ball = 'solid' if game_idx % 2 == 0 else 'stripe'
        env.reset(target_ball=target_ball)
        
        print(f"球型分配: Player A 打 {target_ball}")
        
        # 游戏循环
        shot_count = 0
        max_shots = 100  # 防止无限循环
        
        while shot_count < max_shots:
            done, info = env.get_done()
            if done:
                break
            
            curr_player_idx = env.curr_player
            agent = agents[(curr_player_idx + game_idx) % 2]  # 轮换先后手
            agent_name = agent_names[(curr_player_idx + game_idx) % 2]
            player_str = env.players[curr_player_idx]
            
            print(f"\n[第{env.hit_count}次击球] {agent_name} (玩家 {player_str})")
            
            # 获取观测
            balls, my_targets, table = env.get_observation(player=player_str)
            
            # Agent 决策
            action = agent.decision(
                balls=balls,
                my_targets=my_targets,
                table=table
            )
            
            # 执行动作
            env.take_shot(action)
            shot_count += 1
        
        # 游戏结束
        done, info = env.get_done()
        if not done:
            print(f"\n达到最大击球数 {max_shots}，游戏强制结束")
            continue
        
        winner_name_str = env.winner  # 'A' or 'B' or 'SAME'
        
        if winner_name_str == 'SAME':
            print(f"\n{'='*60}")
            print(f"第 {game_idx + 1} 局结束 - 平局")
            print(f"{'='*60}")
            continue
        
        # 确定实际获胜的Agent（考虑轮换）
        winner_player_idx = 0 if winner_name_str == 'A' else 1
        actual_winner_idx = (winner_player_idx + game_idx) % 2
        winner_name = agent_names[actual_winner_idx]
        wins[actual_winner_idx] += 1
        
        print(f"\n{'='*60}")
        print(f"第 {game_idx + 1} 局结束")
        print(f"获胜者: {winner_name} (玩家 {winner_name_str})")
        print(f"当前战绩: {agent_names[0]} {wins[0]} - {wins[1]} {agent_names[1]}")
        print(f"{'='*60}")
    
    # 最终统计
    print(f"\n{'='*60}")
    print("最终结果")
    print(f"{'='*60}")
    print(f"{agent_names[0]}: {wins[0]} 胜")
    print(f"{agent_names[1]}: {wins[1]} 胜")
    if num_games > 0:
        print(f"HybridAgent 胜率: {wins[0]/num_games*100:.1f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

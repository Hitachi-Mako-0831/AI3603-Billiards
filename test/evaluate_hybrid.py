"""
使用 HybridAgent 的评估脚本示例

修改自 evaluate.py，用于测试 HybridAgent vs BasicAgent
"""

import argparse
from pathlib import Path
from poolenv import PoolEnv
from agent import BasicAgent, HybridAgent

def main():
    parser = argparse.ArgumentParser(description="Evaluate HybridAgent against BasicAgent")
    parser.add_argument("--games", type=int, default=10, help="评估局数")
    parser.add_argument("--use-value-net", action="store_true", help="使用学习型评估器")
    parser.add_argument("--value-net-path", type=str, default=None, help="评估器模型路径")
    args = parser.parse_args()
    
    env = PoolEnv(verbose=True, record_shots=False)
    results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
    n_games = args.games
    
    # Agent A: BasicAgent (对手)
    agent_a = BasicAgent()
    
    # Agent B: HybridAgent (我们的agent)
    agent_b = HybridAgent(
        use_value_network=args.use_value_net,
        value_net_path=args.value_net_path
    )
    
    players = [agent_a, agent_b]  # 用于切换先后手
    target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']  # 轮换球型
    
    for i in range(n_games):
        print()
        print(f"------- 第 {i} 局比赛开始 -------")
        env.reset(target_ball=target_ball_choice[i % 4])
        print(f"本局 Player A: {players[i % 2].__class__.__name__}, 目标球型: {target_ball_choice[i % 4]}")
        
        # 游戏循环
        shot_count = 0
        max_shots = 100
        
        while shot_count < max_shots:
            done, info = env.get_done()
            if done:
                break
            
            player = env.get_curr_player()
            print(f"[第{env.hit_count}次击球] player: {player}")
            
            balls, my_targets, table = env.get_observation(player)
            
            if player == 'A':
                action = players[i % 2].decision(balls, my_targets, table)
            else:
                action = players[(i + 1) % 2].decision(balls, my_targets, table)
            
            step_info = env.take_shot(action)
            
            # 检查犯规
            if step_info.get('FOUL_FIRST_HIT'):
                print("本杆判罚：首次接触对方球或黑8，直接交换球权。")
            if step_info.get('NO_POCKET_NO_RAIL'):
                print("本杆判罚：无进球且母球或目标球未碰库，直接交换球权。")
            if step_info.get('NO_HIT'):
                print("本杆判罚：白球未接触任何球，直接交换球权。")
            
            shot_count += 1
        
        # 统计结果
        done, info = env.get_done()
        if done:
            winner = info['winner']
            results[f'AGENT_{winner}_WIN'] += 1
            print(f"第 {i} 局结束，获胜者: Player {winner}")
        else:
            print(f"第 {i} 局超时")
    
    # 最终统计
    print()
    print("=" * 60)
    print(f"{'='*20} 评估结果 {'='*20}")
    print(f"总局数: {n_games}")
    print(f"BasicAgent 胜场 (A): {results['AGENT_A_WIN']}")
    print(f"HybridAgent 胜场 (B): {results['AGENT_B_WIN']}")
    print(f"平局: {results['SAME']}")
    
    if n_games > 0:
        win_rate = results['AGENT_B_WIN'] / n_games * 100
        print(f"\nHybridAgent 胜率: {win_rate:.1f}%")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

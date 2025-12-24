"""
简化版对战测试 - HybridAgent vs BasicAgent
只打3局快速验证性能
"""

from poolenv import PoolEnv
from agent import BasicAgent, HybridAgent
import time

def main():
    print("=" * 60)
    print("HybridAgent vs BasicAgent 快速对战 (3局)")
    print("=" * 60)
    
    env = PoolEnv(verbose=False, record_shots=False)  # 关闭详细日志
    
    agent_a = HybridAgent(use_value_network=False)
    agent_b = BasicAgent()
    
    agents = [agent_a, agent_b]
    agent_names = ["HybridAgent", "BasicAgent"]
    
    wins = [0, 0]
    
    for game_idx in range(3):
        print(f"\n{'='*60}")
        print(f"第 {game_idx + 1} 局")
        print(f"{'='*60}")
        
        target_ball = 'solid' if game_idx % 2 == 0 else 'stripe'
        env.reset(target_ball=target_ball)
        
        game_start = time.time()
        shot_count = 0
        max_shots = 50  # 限制最大击球数
        
        while shot_count < max_shots:
            done, info = env.get_done()
            if done:
                break
            
            curr_player_idx = env.curr_player
            agent = agents[(curr_player_idx + game_idx) % 2]
            agent_name = agent_names[(curr_player_idx + game_idx) % 2]
            player_str = env.players[curr_player_idx]
            
            print(f"\\n[击球{shot_count+1}] {agent_name} (玩家 {player_str})", end=" ")
            
            balls, my_targets, table = env.get_observation(player=player_str)
            
            shot_start = time.time()
            action = agent.decision(balls=balls, my_targets=my_targets, table=table)
            shot_time = time.time() - shot_start
            
            print(f"- 决策耗时: {shot_time:.1f}秒", end="")
            
            step_info = env.take_shot(action)
            
            # 显示结果
            if step_info.get('pocketed'):
                print(f" ✅ 进球: {step_info['pocketed']}")
            elif step_info.get('FOUL_FIRST_HIT'):
                print(" ❌ 犯规: 首次接触对方球")
            elif step_info.get('NO_POCKET_NO_RAIL'):
                print(" ❌ 犯规: 无进球未碰库")
            else:
                print(" ⚪ 无进球")
            
            shot_count += 1
        
        game_time = time.time() - game_start
        
        done, info = env.get_done()
        if not done:
            print(f"\\n⏱️ 达到最大击球数，游戏超时")
            continue
        
        winner_name_str = env.winner
        
        if winner_name_str == 'SAME':
            print(f"\\n🤝 平局")
            continue
        
        winner_player_idx = 0 if winner_name_str == 'A' else 1
        actual_winner_idx = (winner_player_idx + game_idx) % 2
        winner_name = agent_names[actual_winner_idx]
        wins[actual_winner_idx] += 1
        
        print(f"\\n{'='*60}")
        print(f"🏆 获胜者: {winner_name}")
        print(f"⏱️ 游戏时长: {game_time:.1f}秒, 总击球: {shot_count}次")
        print(f"📊 当前战绩: {agent_names[0]} {wins[0]} - {wins[1]} {agent_names[1]}")
        print(f"{'='*60}")
    
    print(f"\\n{'='*60}")
    print("最终结果")
    print(f"{'='*60}")
    print(f"🤖 {agent_names[0]}: {wins[0]} 胜")
    print(f"🎯 {agent_names[1]}: {wins[1]} 胜")
    if sum(wins) > 0:
        print(f"\\n💪 HybridAgent 胜率: {wins[0]/sum(wins)*100:.1f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

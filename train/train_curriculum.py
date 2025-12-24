"""
train_curriculum.py - 课程学习训练脚本

从简单场景开始，逐步增加难度：
- Level 1: 1个目标球（学习瞄准和击球）
- Level 2: 3个目标球（学习选择目标）
- Level 3: 5个目标球 + 干扰球
- Level 4: 7个目标球 + 干扰球
- Level 5: 完整游戏
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from curriculum_env import CurriculumPoolEnv
from sac import SACAgent, SACConfig

SAFE_ACTION_LIMITS: Dict[str, Tuple[float, float]] = {
    "V0": (0.5, 6.0),
    "phi": (0.0, 360.0),
    "theta": (0.0, 80.0),
    "a": (-0.30, 0.30),
    "b": (-0.30, 0.30),
}


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Shot simulation timed out")


def safe_take_shot(env, action_dict: dict, timeout_sec: float = 10.0):
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    
    try:
        step_info = env.take_shot(action_dict)
        return step_info, False
    except TimeoutError:
        print(f"[Warning] Shot timed out after {timeout_sec}s")
        return None, True
    except Exception as e:
        print(f"[Warning] Shot failed: {e}")
        return None, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def enforce_action_bounds(action: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
    clipped = {}
    changed = False
    for key, (low, high) in SAFE_ACTION_LIMITS.items():
        value = float(action.get(key, low))
        clamped = float(np.clip(value, low, high))
        if abs(clamped - value) > 1e-6:
            changed = True
        clipped[key] = clamped
    return clipped, changed


def parse_args():
    parser = argparse.ArgumentParser(description="Curriculum learning for billiards")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--start-level", type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument("--auto-promote", action="store_true", default=True)
    parser.add_argument("--no-auto-promote", action="store_false", dest="auto_promote")
    parser.add_argument("--promote-threshold", type=float, default=0.25)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sac_curriculum.pth")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--shot-timeout", type=float, default=15.0)
    return parser.parse_args()


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main():
    args = parse_args()
    set_global_seed(args.seed)
    
    # 创建课程学习环境（单人训练模式）
    env = CurriculumPoolEnv(verbose=False, record_shots=False, single_player_mode=True)
    env.set_curriculum_level(args.start_level)
    env.auto_curriculum = args.auto_promote
    env.promotion_threshold = args.promote_threshold
    env.enable_noise = False  # 课程学习初期关闭噪声
    
    checkpoint_base = Path(args.checkpoint)
    
    # SAC配置
    sac_config = SACConfig(
        hidden_dim=args.hidden_dim,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=1e-4,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        automatic_entropy_tuning=True,
        policy_update_freq=2,
        normalize_rewards=True,
    )
    
    sac_agent = SACAgent(
        config=sac_config,
        checkpoint_path=str(checkpoint_base),
        training=True,
    )
    
    # 日志
    log_path = Path(args.log_dir) / "curriculum_training.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'level', 'reward', 'turns', 'pocketed', 
                        'success', 'buffer_size', 'total_updates'])
    
    total_updates = 0
    start_time = time.time()
    
    # 统计
    level_rewards = {i: [] for i in range(1, 6)}
    level_success = {i: [] for i in range(1, 6)}
    
    print(f"🎓 课程学习训练开始")
    print(f"   起始难度: Level {args.start_level}")
    print(f"   自动升级: {'开启' if args.auto_promote else '关闭'}")
    print(f"   升级阈值: {args.promote_threshold*100:.0f}%")
    print("="*60)
    
    for episode in range(1, args.episodes + 1):
        current_level = env.get_curriculum_level()
        target_ball = random.choice(['solid', 'stripe'])
        env.reset(target_ball=target_ball)
        
        episode_reward = 0.0
        agent_turns = 0
        total_pocketed = 0
        done = False
        aborted = False
        
        while not done and not aborted:
            # 获取观测（单人模式，始终是玩家A）
            balls, my_targets, table = env.get_observation("A", copy_state=False)
            state = sac_agent.encode_observation(balls, my_targets, table)
            
            # 选择动作
            action_dict, scaled_action = sac_agent._act(state, evaluate=False)
            action_dict, _ = enforce_action_bounds(action_dict)
            
            # 执行
            step_info, timed_out = safe_take_shot(env, action_dict, args.shot_timeout)
            
            if timed_out or step_info is None:
                aborted = True
                break
            
            # 计算奖励
            reward = SACAgent.compute_dense_reward(step_info, my_targets)
            
            # 课程学习额外奖励：进球奖励加成（控制奖励尺度）
            pocketed = len(step_info.get('ME_INTO_POCKET', []))
            total_pocketed += pocketed
            if pocketed > 0:
                # 使用对数增长，避免高级别奖励过大
                level_bonus = 1 + math.log(current_level + 1, 2)  # Level 1-5 对应 1.0-2.58
                reward += 30 * pocketed * level_bonus  # Level 5最多 +77.4 每球
            
            # 获取下一状态
            next_balls, next_targets, next_table = env.get_observation("A", copy_state=False)
            next_state = sac_agent.encode_observation(next_balls, next_targets, next_table)
            
            done, info = env.get_done()
            
            # 存储经验（传入动作字典，不是scaled_action）
            sac_agent.store_transition(state, action_dict, reward, next_state, done)
            episode_reward += reward
            agent_turns += 1
            
            # 更新网络
            if len(sac_agent.replay_buffer) >= 1024:
                for _ in range(2):
                    update_info = sac_agent.update_parameters()
                    if update_info:
                        total_updates += 1
        
        # 判断是否成功
        success = (env.winner == "A") if env.done else False
        
        # 记录统计
        level_rewards[current_level].append(episode_reward)
        level_success[current_level].append(1 if success else 0)
        
        # 写入日志
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, current_level, f"{episode_reward:.1f}", 
                           agent_turns, total_pocketed, int(success),
                           len(sac_agent.replay_buffer), total_updates])
        
        # 打印进度
        if episode % 50 == 0 or env.curriculum_level != current_level:
            elapsed = time.time() - start_time
            
            # 计算当前级别统计
            recent_rewards = level_rewards[current_level][-100:]
            recent_success = level_success[current_level][-100:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            success_rate = np.mean(recent_success) if recent_success else 0
            
            print(f"[Ep {episode:5d}] Level={env.curriculum_level} | "
                  f"Reward={avg_reward:7.1f} | Success={success_rate*100:5.1f}% | "
                  f"Pocketed={total_pocketed} | Buffer={len(sac_agent.replay_buffer)} | "
                  f"Time={elapsed/60:.1f}min")
        
        # 保存检查点
        if episode % args.save_every == 0:
            checkpoint_path = checkpoint_base.parent / f"sac_curriculum_L{env.curriculum_level}_ep{episode}.pth"
            sac_agent.save_checkpoint(output_path=checkpoint_path)
            sac_agent.save_checkpoint()  # 同时保存默认路径
            print(f"💾 保存检查点: {checkpoint_path}")
        
        # 如果达到最高级别且表现稳定，可以提前结束
        if env.curriculum_level == 5:
            recent = level_success[5][-200:]
            if len(recent) >= 200 and np.mean(recent) >= 0.4:
                print(f"🎉 Level 5 达到稳定表现，训练完成!")
                break
    
    # 最终统计
    print("\n" + "="*60)
    print("📊 课程学习训练结束")
    print("="*60)
    
    for level in range(1, 6):
        rewards = level_rewards[level]
        successes = level_success[level]
        if rewards:
            print(f"Level {level}: Episodes={len(rewards)}, "
                  f"AvgReward={np.mean(rewards):.1f}, "
                  f"SuccessRate={np.mean(successes)*100:.1f}%")
    
    # 保存最终模型
    sac_agent.save_checkpoint()
    print(f"\n✅ 最终模型已保存: {checkpoint_base}")


if __name__ == "__main__":
    main()

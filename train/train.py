from __future__ import annotations

import argparse
import csv
import random
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

# 抑制 pooltool 内部的数值警告
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, ROOT_DIR):
	if str(path) not in sys.path:
		sys.path.append(str(path))

from poolenv import PoolEnv  
from sac import SACAgent, SACConfig  
from agent import BasicAgent

SAFE_ACTION_LIMITS: Dict[str, Tuple[float, float]] = {
	"V0": (0.5, 6.0),       
	"phi": (0.0, 360.0),
	"theta": (0.0, 80.0),   
	"a": (-0.30, 0.30),    
	"b": (-0.30, 0.30),
}


class TimeoutError(Exception):
	"""自定义超时异常"""
	pass


def timeout_handler(signum, frame):
	raise TimeoutError("Shot simulation timed out")


def safe_take_shot(env: PoolEnv, action_dict: dict, timeout_sec: float = 10.0):
	"""
	带超时保护的击球函数，防止物理模拟陷入死循环。
	
	返回:
		step_info: 击球结果，如果超时则返回 None
		timed_out: 是否超时
	"""
	# 设置信号处理器
	old_handler = signal.signal(signal.SIGALRM, timeout_handler)
	signal.setitimer(signal.ITIMER_REAL, timeout_sec)
	
	try:
		step_info = env.take_shot(action_dict)
		return step_info, False
	except TimeoutError:
		print(f"[Warning] Shot timed out after {timeout_sec}s, skipping...")
		return None, True
	except Exception as e:
		print(f"[Warning] Shot failed with error: {e}")
		return None, True
	finally:
		# 取消计时器并恢复原来的处理器
		signal.setitimer(signal.ITIMER_REAL, 0)
		signal.signal(signal.SIGALRM, old_handler)


def enforce_action_bounds(action: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
	"""Clip action dict to conservative bounds to avoid unstable physics."""
	clipped: Dict[str, float] = {}
	changed = False
	for key, (low, high) in SAFE_ACTION_LIMITS.items():
		value = float(action.get(key, low))
		clamped = float(np.clip(value, low, high))
		if abs(clamped - value) > 1e-6:
			changed = True
		clipped[key] = clamped
	return clipped, changed


def values_are_finite(name: str, value) -> bool:
	arr = np.asarray(value, dtype=np.float32)
	if np.all(np.isfinite(arr)):
		return True
	return False


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Soft Actor-Critic training for AI3603 billiards")
	parser.add_argument("--episodes", type=int, default=200, help="the number of episode")
	parser.add_argument("--opponent", type=str, choices=["base", "sac", "none"], default="base", help="the opponent player type (none=solo practice)")
	parser.add_argument("--control-player", type=str, choices=["A", "B"], default="A", help="the player controled by SAC agent")
	parser.add_argument("--target-cycle", type=str, default="solid,stripe", help="recyclable ball type, split by comma")
	parser.add_argument("--checkpoint", type=str, default="checkpoints/sac_agent.pth", help="checkpoint path")
	parser.add_argument("--log-dir", type=str, default="logs", help="log path")
	parser.add_argument("--save-every", type=int, default=20, help="how many echos we save the model")
	parser.add_argument("--selfplay-sync", type=int, default=50, help="episodes between syncing opponent weights")
	parser.add_argument("--seed", type=int, default=42, help="randon seed")
	parser.add_argument("--env-noise", action="store_true", help="use the environment noise")
	parser.add_argument("--learning-starts", type=int, default=1024, help="start gradient update when buffer size reach this value")
	parser.add_argument("--updates-per-step", type=int, default=4, help="update times of the agent in a episodes")
	parser.add_argument("--hidden-dim", type=int, default=512, help="Actor/Critic hidden layer dimension")
	parser.add_argument("--batch-size", type=int, default=256, help="SAC batch size")
	parser.add_argument("--buffer-size", type=int, default=500_000, help="buffer size")
	parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
	parser.add_argument("--tau", type=float, default=0.005, help="target network update soft factor")
	parser.add_argument("--lr-actor", type=float, default=1e-4, help="Actor learning rate")
	parser.add_argument("--lr-critic", type=float, default=1e-4, help="Critic learning rate")
	parser.add_argument("--lr-alpha", type=float, default=1e-4, help="alpha learning rate")
	parser.add_argument("--alpha", type=float, default=0.2, help="solid alpha (when forbidden auto entroy adjust)")
	parser.add_argument("--disable-auto-entropy", action="store_true", help="diable automatic entropy adjustment")
	parser.add_argument("--policy-update-freq", type=int, default=5, help="frequency of actor/alpha updates vs critic updates")
	parser.add_argument("--shot-timeout-sec", type=float, default=20.0, help="abort current episode if a single take_shot exceeds this wallclock time")
	parser.add_argument("--no-reward-norm", action="store_true", help="disable reward normalization")
	return parser.parse_args()


def set_global_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	try:
		import torch
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	except ImportError:
		pass


def opponent_action(opponent_agent, balls, my_targets, table):
	"""为对手选择动作，支持 SAC 自博弈或基础启发式 Agent。"""
	if isinstance(opponent_agent, SACAgent):
		state = opponent_agent.encode_observation(balls, my_targets, table)
		action_dict, _ = opponent_agent._act(state, evaluate=True)
	else:
		action_dict = opponent_agent.decision(balls=balls, my_targets=my_targets, table=table)
	return action_dict


def rollout_opponent_turns(env: PoolEnv, opponent_agent, control_player: str, timeout_sec: float = 10.0) -> Tuple[float, bool, bool]:
	"""让对手智能体连续出杆直到轮到训练智能体或对局结束
	
	返回:
		cumulative_penalty: 累计惩罚
		done: 游戏是否结束
		timed_out: 是否因超时而中断
	"""
	cumulative_penalty = 0.0
	done, _ = env.get_done()
	while not done and env.get_curr_player() != control_player:
		player = env.get_curr_player()
		balls, my_targets, table = env.get_observation(player, copy_state=False)
		action_dict = opponent_action(opponent_agent, balls, my_targets, table)
		action_dict, _ = enforce_action_bounds(action_dict)
		step_info, timed_out = safe_take_shot(env, action_dict, timeout_sec)
		if timed_out or step_info is None:
			return cumulative_penalty, True, True  # 超时视为对局结束
		cumulative_penalty -= SACAgent.compute_dense_reward(step_info, my_targets)
		done, _ = env.get_done()
	return cumulative_penalty, done, False


def sync_opponent_agent(source: SACAgent, target: SACAgent) -> None:
    """同步对手的参数"""
    target.actor.load_state_dict(source.actor.state_dict())
    target.actor.eval()


def format_checkpoint_variant(base_path: Path, tag: str) -> Path:
    """生成checkpoint文件名"""
    return base_path.with_name(f"{base_path.stem}_{tag}{base_path.suffix}")


def append_metrics(log_path: Path, row: dict) -> None:
    """添加日志信息"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    fieldnames = [
		"episode",
		"reward",
		"agent_turns",
		"buffer_size",
		"total_updates",
		"total_env_steps",
		"elapsed_sec",
	]
    with log_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    # 读取参数
	args = parse_args()
	set_global_seed(args.seed)

	target_cycle = [item.strip() for item in args.target_cycle.split(",") if item.strip()]
	if not target_cycle:
		raise ValueError("target-cycle 至少需要一个取值 (solid/stripe)")
	if args.save_every <= 0:
		raise ValueError("--save-every must be positive")
	if args.selfplay_sync <= 0:
		raise ValueError("--selfplay-sync must be positive")

	# 生成训练环境
	env = PoolEnv(verbose=False, record_shots=False)
	env.enable_noise = args.env_noise
	checkpoint_base = Path(args.checkpoint)

	# 训练Agent参数
	sac_config = SACConfig(
		hidden_dim=args.hidden_dim,
		gamma=args.gamma,
		tau=args.tau,
		alpha=args.alpha,
		lr_actor=args.lr_actor,
		lr_critic=args.lr_critic,
		lr_alpha=args.lr_alpha,
		batch_size=args.batch_size,
		buffer_size=args.buffer_size,
		automatic_entropy_tuning=not args.disable_auto_entropy,
		policy_update_freq=args.policy_update_freq,
		normalize_rewards=not args.no_reward_norm,  # 默认开启奖励归一化
	)

	sac_agent = SACAgent(
		config=sac_config,
		checkpoint_path=str(checkpoint_base),
		training=True,
	)
	
	# 对手Agent（none模式下无对手，agent独自练习）
	opponent_agent = None
	solo_mode = (args.opponent == "none")
	
	if args.opponent == "base":
		opponent_agent = BasicAgent() 
		opponent_agent.enable_noise = args.env_noise
	elif args.opponent == "sac":
		opponent_agent = SACAgent(
			config=sac_config,
			checkpoint_path=str(format_checkpoint_variant(checkpoint_base, "opponent")),
			training=False,
		)
		sync_opponent_agent(sac_agent, opponent_agent)
		opponent_agent.enable_noise = args.env_noise
	# opponent == "none" 时，opponent_agent 保持 None
  
	log_path = Path(args.log_dir) / "training_metrics.csv"

	total_env_steps = 0
	total_updates = 0
	start_time = time.time()

	for episode in range(1, args.episodes + 1):
		target_ball = target_cycle[(episode - 1) % len(target_cycle)]
		env.reset(target_ball=target_ball)
		env.enable_noise = args.env_noise

		episode_reward = 0.0
		agent_turns = 0
		aborted_episode = False
		done, _ = env.get_done()

		# 如果对手先手，对手先行动（单人模式跳过）
		if not solo_mode and env.get_curr_player() != args.control_player:
			penalty, done, timed_out = rollout_opponent_turns(env, opponent_agent, args.control_player, args.shot_timeout_sec)
			episode_reward += penalty
			if timed_out:
				aborted_episode = True

		# 游戏主循环
		while not done and not aborted_episode:
			# 确保轮到训练智能体（单人模式跳过对手回合）
			if not solo_mode and env.get_curr_player() != args.control_player:
				penalty, done, timed_out = rollout_opponent_turns(env, opponent_agent, args.control_player, args.shot_timeout_sec)
				episode_reward += penalty
				if timed_out:
					aborted_episode = True
					break
				if done:
					break
			
			# 获取当前状态
			# 单人模式：始终使用当前击球方的视角（环境内部会切换）
			# 对战模式：使用控制玩家的视角
			if solo_mode:
				player_for_obs = env.get_curr_player()  # 跟随环境的当前玩家
			else:
				player_for_obs = args.control_player
			balls, my_targets, table = env.get_observation(player_for_obs, copy_state=False)
			state = sac_agent.encode_observation(balls, my_targets, table)
			if not values_are_finite("state", state):
				aborted_episode = True
				break

			# 选择动作
			action_dict, _ = sac_agent._act(state, evaluate=False)
			action_dict, clipped = enforce_action_bounds(action_dict)
			if not values_are_finite("action", list(action_dict.values())):
				aborted_episode = True
				break

			# 执行动作（带超时保护）
			step_info, timed_out = safe_take_shot(env, action_dict, args.shot_timeout_sec)
			if timed_out or step_info is None:
				aborted_episode = True
				break
			immediate_reward = SACAgent.compute_dense_reward(step_info, my_targets)
			if not values_are_finite("reward", immediate_reward):
				aborted_episode = True
				break
			total_env_steps += 1

			# 检查游戏是否结束
			done, _ = env.get_done()
			opponent_penalty = 0.0
			if not solo_mode and not done:
				opponent_penalty, done, timed_out = rollout_opponent_turns(env, opponent_agent, args.control_player, args.shot_timeout_sec)
				if timed_out:
					aborted_episode = True
					break

			# 计算总奖励
			total_reward = immediate_reward + opponent_penalty
			if not values_are_finite("total_reward", total_reward):
				aborted_episode = True
				break
			episode_reward += total_reward

			# 获取下一个状态
			if done:
				next_state = state  # 保持当前状态，(1-dones)会清零Q值贡献
			else:
				next_player = player_for_obs
				next_balls, next_targets, next_table = env.get_observation(next_player, copy_state=False)
				next_state = sac_agent.encode_observation(next_balls, next_targets, next_table)
				if not values_are_finite("next_state", next_state):
					aborted_episode = True
					break

			# 存储转移
			sac_agent.store_transition(state, action_dict, total_reward, next_state, done)

			# 更新网络参数
			if sac_agent.replay_buffer and len(sac_agent.replay_buffer) >= args.learning_starts:
				for _ in range(args.updates_per_step):
					update_info = sac_agent.update_parameters()
					if update_info is not None:
						total_updates += 1

			agent_turns += 1

		# episode结束处理
		if aborted_episode:
			print(f"[Safety] Episode {episode} aborted due to invalid physics state; skipping remaining shots.")

		# 定期保存checkpoint
		if episode % args.save_every == 0:
			checkpoint_path = format_checkpoint_variant(checkpoint_base, f"ep{episode}")
			sac_agent.save_checkpoint(checkpoint_path)

		# 自博弈时定期同步对手
		if opponent_agent is not None and isinstance(opponent_agent, SACAgent) and episode % args.selfplay_sync == 0:
			sync_opponent_agent(sac_agent, opponent_agent)

		# 记录日志
		elapsed = time.time() - start_time
		buffer_size = len(sac_agent.replay_buffer) if sac_agent.replay_buffer else 0
		mode_str = "solo" if solo_mode else args.opponent
		append_metrics(
			log_path,
			{
				"episode": episode,
				"reward": round(episode_reward, 2),
				"agent_turns": agent_turns,
				"buffer_size": buffer_size,
				"total_updates": total_updates,
				"total_env_steps": total_env_steps,
				"elapsed_sec": round(elapsed, 2),
			},
		)

		# 打印训练进度
		print(
			f"[Episode {episode}/{args.episodes}] reward={episode_reward:.1f} turns={agent_turns} "
			f"buffer={buffer_size} updates={total_updates} target={target_ball}"
		)

	# 训练结束，保存最终模型
	sac_agent.save_checkpoint()
	print("Training finished. Checkpoints saved to", args.checkpoint)


if __name__ == "__main__":
	main()

"""
统一训练入口
"""

import argparse
import yaml
import os
import time
from algorithms import get_algorithm_class
from envs.wrapper import BilliardGymEnv
try:
    from agent import BasicAgent
except ImportError:
    print("[!] Could not import BasicAgent. Make sure agent.py is in the path.")
    BasicAgent = None

def main():
    parser = argparse.ArgumentParser(description="Billiards RL Training Manager")
    parser.add_argument("--algo", type=str, default="random", help="Algorithm name (e.g., random, ppo, sac)")
    parser.add_argument("--config", type=str, default="", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="test", choices=["test", "train"], help="Running mode")
    parser.add_argument("--enable_opponent_noise", action="store_true", help="Enable noise for BasicAgent opponent")
    parser.add_argument("--opponent_noise_scale", type=float, default=1.0, help="Scale factor for opponent noise standard deviation")
    args = parser.parse_args()

    # 1. 加载配置
    print(f"[*] Starting {args.mode} with algorithm: {args.algo}")
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        print("[!] No config file provided, using default settings.")
        # Default PPO config if not provided
        if args.algo == 'ppo':
            cfg = {
                'lr': 0.0003,
                'gamma': 0.99,
                'k_epochs': 4,
                'eps_clip': 0.2,
                'max_timesteps': 100000,
                'update_timestep': 2000,
                'save_interval': 10000
            }

    # 2. 初始化环境
    print("[*] Initializing environment...")
    
    opponent_agent = None
    if BasicAgent:
        print("[*] Initializing BasicAgent as opponent...")
        try:
            opponent_agent = BasicAgent()
            # Configure opponent noise based on arguments
            if args.enable_opponent_noise:
                opponent_agent.enable_noise = True
                for k in opponent_agent.noise_std:
                    opponent_agent.noise_std[k] *= args.opponent_noise_scale
                print(f"[*] Opponent noise ENABLED with scale {args.opponent_noise_scale}")
                print(f"    Noise params: {opponent_agent.noise_std}")
            else:
                opponent_agent.enable_noise = False
                print("[*] Opponent noise DISABLED")
        except Exception as e:
            print(f"[!] Failed to initialize BasicAgent: {e}")
            opponent_agent = None
            
    env = BilliardGymEnv(env_config=cfg.get("env", {}), opponent=opponent_agent)
    obs, _ = env.reset(seed=args.seed)
    print(f"    Observation Space: {env.observation_space.shape}")
    print(f"    Action Space: {env.action_space.shape}")

    # 3. 初始化算法
    print(f"[*] Initializing agent: {args.algo}...")
    try:
        AgentClass = get_algorithm_class(args.algo)
        agent = AgentClass(cfg)
    except ValueError as e:
        print(f"[x] Error: {e}")
        return

    # 4. 执行循环
    if args.mode == 'test':
        print("[*] Starting sanity check loop (10 steps)...")
        try:
            for i in range(10):
                action = agent.select_action(obs, evaluate=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"    Step {i+1}: Action={action[:2]}... | Reward={reward:.2f}")
                
                if terminated or truncated:
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                    
            print("[√] Sanity check passed! The architecture is working.")
            
        except Exception as e:
            print(f"[x] Runtime Error: {e}")
            import traceback
            traceback.print_exc()
            
    elif args.mode == 'train':
        print("[*] Starting training loop...")
        
        max_timesteps = cfg.get('max_timesteps', 100000)
        update_timestep = cfg.get('update_timestep', 2000)
        save_interval = cfg.get('save_interval', 10000)
        
        time_step = 0
        i_episode = 0
        
        try:
            while time_step < max_timesteps:
                # Rotation logic per GAME_RULES.md
                # Game 0: Agent=A(Solid), Opponent=B(Stripe)
                # Game 1: Agent=B(Stripe), Opponent=A(Solid) -> Agent is Player B
                # Game 2: Agent=A(Stripe), Opponent=B(Solid)
                # Game 3: Agent=B(Solid), Opponent=A(Stripe) -> Agent is Player B
                
                rotation_idx = i_episode % 4
                reset_options = {}
                
                if rotation_idx == 0:
                    reset_options = {'agent_id': 'A', 'target_ball': 'solid'}
                elif rotation_idx == 1:
                    # We are Player B. Opponent is Player A (solid).
                    reset_options = {'agent_id': 'B', 'target_ball': 'solid'}
                elif rotation_idx == 2:
                    reset_options = {'agent_id': 'A', 'target_ball': 'stripe'}
                elif rotation_idx == 3:
                    # We are Player B. Opponent is Player A (stripe).
                    reset_options = {'agent_id': 'B', 'target_ball': 'stripe'}
                    
                state, _ = env.reset(options=reset_options)
                current_ep_reward = 0
                
                for t in range(1000): # Max episode length safety
                    time_step += 1
                    
                    if time_step % 10 == 0:
                        print(f"    [Step {time_step}] Simulating...", end='\r')

                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    
                    # Save data for PPO (or other on-policy algos)
                    if hasattr(agent, 'store_transition'):
                        agent.store_transition(reward, terminated or truncated)
                    
                    state = next_state
                    current_ep_reward += reward
                    
                    # Update Agent
                    if hasattr(agent, 'train') and time_step % update_timestep == 0:
                        print(f"    [Train] Updating agent at step {time_step}...")
                        agent.train()
                        
                    # Save Model
                    if time_step % save_interval == 0:
                        print(f"    [Save] Saving model at step {time_step}...")
                        if not os.path.exists('checkpoints'):
                            os.makedirs('checkpoints')
                        agent.save(f"checkpoints/{args.algo}_{time_step}.pth")
                        
                    if terminated or truncated:
                        break
                
                i_episode += 1
                print(f"Episode {i_episode} | Total Steps: {time_step} | Reward: {current_ep_reward:.2f}")

            # Save final model
            print(f"[*] Training finished. Saving final model...")
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            agent.save(f"checkpoints/{args.algo}_final.pth")
                
        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user.")
            print(f"[*] Saving model before exit...")
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            agent.save(f"checkpoints/{args.algo}_interrupted.pth")
            
        except Exception as e:
            print(f"[x] Training Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

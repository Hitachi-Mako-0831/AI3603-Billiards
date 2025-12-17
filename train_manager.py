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

class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        self.headers = ['Episode', 'Step', 'Reward', 'ActorLoss', 'CriticLoss', 'TotalLoss']
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(','.join(self.headers) + '\n')
                
    def log(self, episode, step, reward, metrics=None):
        metrics = metrics or {}
        row = [
            str(episode),
            str(step),
            f"{reward:.4f}",
            f"{metrics.get('actor_loss', 0):.6f}",
            f"{metrics.get('critic_loss', 0):.6f}",
            f"{metrics.get('loss', 0):.6f}"
        ]
        with open(self.filename, 'a') as f:
            f.write(','.join(row) + '\n')

    def get_last_episode(self):
        """Read last line to get resume episode"""
        if not os.path.exists(self.filename):
            return 0
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip().split(',')
                    return int(last_line[0])
        except:
            return 0
        return 0

    def get_last_metrics(self):
        """Read last line to get resume metrics (loss)"""
        if not os.path.exists(self.filename):
            return None
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip().split(',')
                    # headers = ['Episode', 'Step', 'Reward', 'ActorLoss', 'CriticLoss', 'TotalLoss']
                    # indices: 0, 1, 2, 3, 4, 5
                    return {
                        'actor_loss': float(last_line[3]),
                        'critic_loss': float(last_line[4]),
                        'loss': float(last_line[5])
                    }
        except:
            return None
        return None

def main():
    parser = argparse.ArgumentParser(description="Billiards RL Training Manager")
    parser.add_argument("--algo", type=str, default="random", help="Algorithm name (e.g., random, ppo, sac)")
    parser.add_argument("--config", type=str, default="", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="test", choices=["test", "train"], help="Running mode")
    parser.add_argument("--load_model", type=str, default="", help="Path to checkpoint file to load")
    parser.add_argument("--max_timesteps", type=int, default=None, help="Maximum training timesteps (overrides config)")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum training episodes")
    parser.add_argument("--update_timestep", type=int, default=None, help="Timesteps between updates")
    parser.add_argument("--update_episode", type=int, default=None, help="Episodes between updates (overrides update_timestep)")
    parser.add_argument("--enable_opponent_noise", action="store_true", help="Enable noise for BasicAgent opponent")
    parser.add_argument("--opponent_noise_scale", type=float, default=1.0, help="Scale factor for opponent noise standard deviation")
    parser.add_argument("--fast_opponent", action="store_true", help="Use faster (but weaker) settings for BasicAgent to speed up training")
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

    # Override max_timesteps if provided in args
    if args.max_timesteps is not None:
        cfg['max_timesteps'] = args.max_timesteps
        print(f"[*] Overriding max_timesteps to {args.max_timesteps}")

    # Allow update_timestep override
    if args.update_timestep is not None:
        cfg['update_timestep'] = args.update_timestep
        print(f"[*] Overriding update_timestep to {args.update_timestep}")

    # Allow update_episode override
    if args.update_episode is not None:
        cfg['update_episode'] = args.update_episode
        print(f"[*] Overriding update to EPISODE mode: every {args.update_episode} episodes")


    # 2. 初始化环境
    print("[*] Initializing environment...")
    
    opponent_agent = None
    if BasicAgent:
        print("[*] Initializing BasicAgent as opponent...")
        try:
            # Determine opponent search settings
            if args.fast_opponent:
                init_search = 5
                opt_search = 2
                print(f"[*] FAST MODE ENABLED: BasicAgent search reduced to {init_search}+{opt_search}")
            else:
                init_search = 20
                opt_search = 10
                
            opponent_agent = BasicAgent(initial_search=init_search, opt_search=opt_search)
            
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
    start_episode = 0
    start_timestep = 0
    
    # Initialize logger
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = CSVLogger(os.path.join(log_dir, f"{args.algo}_training_log.csv"))
    
    try:
        AgentClass = get_algorithm_class(args.algo)
        agent = AgentClass(cfg)
        
        # Load model if requested
        if args.load_model:
            if os.path.exists(args.load_model):
                print(f"[*] Loading model from {args.load_model}...")
                extra_info = agent.load(args.load_model)
                
                # Try to recover training state
                if isinstance(extra_info, dict):
                    start_episode = extra_info.get('episode', 0)
                    start_timestep = extra_info.get('timestep', 0)
                    print(f"[*] Resumed from Episode {start_episode}, Step {start_timestep}")

                # Fallback to log file if not in checkpoint
                if start_episode == 0:
                    last_ep = logger.get_last_episode()
                    if last_ep > 0:
                        start_episode = last_ep + 1
                        print(f"[*] Resumed episode count from log: {start_episode}")
                        
                # Try to recover metrics from log file to avoid zero loss display
                if start_episode > 0:
                    try:
                        last_metrics = logger.get_last_metrics()
                        if last_metrics:
                            # headers = ['Episode', 'Step', 'Reward', 'ActorLoss', 'CriticLoss', 'TotalLoss']
                            # We map them back to train_metrics structure
                            print(f"[*] Recovered metrics from log: {last_metrics}")
                            # Construct a dummy metrics dict with recovered values
                            train_metrics = last_metrics
                    except Exception as e:
                        print(f"[!] Failed to recover metrics from log: {e}")
            else:
                print(f"[!] Model file not found: {args.load_model}")
                # We don't exit here, just continue with random weights or exit? 
                # Better to warn and continue, or exit if user expected it to work.
                # Let's exit to be safe.
                print("[!] Exiting because model file was not found.")
                return
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
        update_episode = cfg.get('update_episode', None)
        save_interval = cfg.get('save_interval', 10000)
        
        # Use provided args if available, else use config, else default
        max_episodes = args.max_episodes if args.max_episodes else cfg.get('max_episodes', 1000000)
        
        time_step = start_timestep
        i_episode = start_episode
        train_metrics = {} # Initialize globally, not per episode

        try:
            while i_episode < max_episodes:
                # Also check max_timesteps if specified
                if args.max_timesteps and time_step >= args.max_timesteps:
                    print(f"[*] Reached max timesteps {args.max_timesteps}. Stopping.")
                    break
                    
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
                # train_metrics = {}  <-- Remove this line (don't clear metrics at start of episode)
                
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
                    
                    # Update Agent (Timestep based)
                    if update_episode is None:
                        if hasattr(agent, 'train') and time_step % update_timestep == 0:
                            print(f"\n{'='*60}")
                            print(f"🚀 [TRAIN] Updating agent network at Step {time_step}...")
                            metrics = agent.train()
                            if metrics:
                                train_metrics = metrics
                                print(f"    📊 Loss: {metrics.get('loss', 0):.4f} "
                                      f"(Actor: {metrics.get('actor_loss', 0):.4f}, "
                                      f"Critic: {metrics.get('critic_loss', 0):.4f})")
                            print(f"{'='*60}\n")
                        
                    # Save Model
                    if time_step % save_interval == 0:
                        print(f"\n{'#'*60}")
                        print(f"💾 [SAVE] Saving checkpoint at Step {time_step}...")
                        if not os.path.exists('checkpoints'):
                            os.makedirs('checkpoints')
                        
                        save_info = {'episode': i_episode, 'timestep': time_step}
                        agent.save(f"checkpoints/{args.algo}_{time_step}.pth", extra_info=save_info)
                        print(f"{'#'*60}\n")
                        
                    if terminated or truncated:
                        break
                
                i_episode += 1
                
                # Update Agent (Episode based)
                if update_episode is not None and i_episode > 0 and i_episode % update_episode == 0:
                     if hasattr(agent, 'train'):
                        print(f"\n{'='*60}")
                        print(f"🚀 [TRAIN] Updating agent network at Episode {i_episode}...")
                        metrics = agent.train()
                        if metrics:
                            train_metrics = metrics
                            print(f"    📊 Loss: {metrics.get('loss', 0):.4f} "
                                  f"(Actor: {metrics.get('actor_loss', 0):.4f}, "
                                  f"Critic: {metrics.get('critic_loss', 0):.4f})")
                        print(f"{'='*60}\n")

                # Log metrics
                # Ensure we have valid metrics before logging. 
                # If train_metrics is empty (e.g. just resumed and haven't trained yet), 
                # try to keep the recovered values or use zeros if absolutely necessary.
                # The issue is that train_metrics might be overwritten by empty dict if we are not careful?
                # Actually train_metrics is global-ish in this loop.
                # But let's make sure we don't log zeros if we have recovered data.
                
                # If we just resumed, train_metrics should have data from get_last_metrics()
                # If we trained, it has new data.
                # If we haven't trained yet (e.g. episode 1-19), it holds the old data (which is correct, as loss hasn't changed).
                
                logger.log(i_episode, time_step, current_ep_reward, train_metrics)
                
                print(f"✅ Episode {i_episode} Finished | Total Steps: {time_step} | Reward: {current_ep_reward:.2f}")
                if train_metrics:
                     print(f"   📈 Latest Loss: {train_metrics.get('loss', 0):.4f}")


            # Save final model
            print(f"[*] Training finished. Saving final model...")
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            
            save_info = {'episode': i_episode, 'timestep': time_step}
            agent.save(f"checkpoints/{args.algo}_final.pth", extra_info=save_info)
                
        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user.")
            print(f"[*] Saving model before exit...")
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
                
            save_info = {'episode': i_episode, 'timestep': time_step}
            agent.save(f"checkpoints/{args.algo}_interrupted.pth", extra_info=save_info)
            
        except Exception as e:
            print(f"[x] Training Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

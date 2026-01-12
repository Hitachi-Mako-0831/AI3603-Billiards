"""
统一训练入口 (Simplified)
"""

import argparse
import yaml
import os
import time
from envs.wrapper import BilliardGymEnv
try:
    from agents.basic_agent import BasicAgent
except ImportError:
    print("[!] Could not import BasicAgent.")
    BasicAgent = None

from agents.new_agent import NewAgent
from torch.utils.tensorboard import SummaryWriter
import random

def main():
    parser = argparse.ArgumentParser(description="Pool Training (Simplified)")
    parser.add_argument('--train', action='store_true', help='Train mode')
    parser.add_argument('--exp_name', type=str, default='sac_simple', help='Experiment name')
    parser.add_argument('--load_model', action='store_true', help='Load latest checkpoint')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--self_play', action='store_true', help='Self play mode (Agent vs Agent)')
    args = parser.parse_args()

    # Configuration (Hardcoded for Simplicity)
    cfg = {
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'batch_size': 512,
        'hidden_size': 256,
        'lr': 0.0003,
        'start_steps': 20000,
        'updates_per_step': 1,
        'target_update_interval': 1,
        'replay_size': 1000000,
        'seed': 42,
        'reward_scale': 0.1,
        'automatic_entropy_tuning': True
    }
    
    # 1. Setup Environment
    env_config = {'render': args.render}
    
    # Initialize Agent
    agent = NewAgent(cfg)
    
    # Load Model if requested
    if args.load_model:
        checkpoint_path = f"checkpoints/{args.exp_name}_final.pth"
        # Also check for latest timestamped one if final doesn't exist?
        # For simplicity, let's look for any matching file in checkpoints/
        if os.path.exists("checkpoints"):
            files = [f for f in os.listdir("checkpoints") if f.startswith(args.exp_name) and f.endswith(".pth")]
            if files:
                # Sort by modification time
                files.sort(key=lambda x: os.path.getmtime(os.path.join("checkpoints", x)))
                latest = files[-1]
                agent.load(os.path.join("checkpoints", latest))
    
    # Opponent Setup
    if args.self_play:
        opponent = agent # Self play
        print("[Env] Self Play Mode: Agent vs Agent")
    else:
        opponent = BasicAgent() if BasicAgent else None
        print("[Env] Standard Mode: Agent vs BasicAgent")

    env = BilliardGymEnv(env_config=env_config, opponent=opponent)

    # Tensorboard
    log_dir = f"logs/{args.exp_name}"
    writer = SummaryWriter(log_dir)

    # Training Loop
    if args.train:
        print(f"[*] Starting Training: {args.exp_name}")
        
        i_episode = 0
        time_step = agent.total_numsteps # Recover step count
        
        # Max steps
        max_steps = 1000000
        
        while time_step < max_steps:
            state, _ = env.reset()
            current_ep_reward = 0
            train_metrics = {}
            
            # Opponent is handled by wrapper if opponent is set.
            # If self_play, wrapper uses 'opponent' (which is agent) to make moves for the other player.
            # So we just step.
            
            while True:
                time_step += 1
                
                if time_step % 10 == 0:
                    print(f"    [Step {time_step}] Simulating...", end='\r')

                # Select Action
                action = agent.select_action(state)
                
                # Step
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Store
                agent.store_transition(state, action, reward, next_state, terminated or truncated)
                
                state = next_state
                current_ep_reward += reward
                
                # Train
                if time_step > agent.start_steps:
                    metrics = agent.train()
                    if metrics:
                        train_metrics = metrics
                
                # Save
                if time_step % 10000 == 0:
                    print(f"\n💾 [SAVE] Saving checkpoint at Step {time_step}...")
                    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                    agent.save(f"checkpoints/{args.exp_name}_{time_step}.pth")

                if terminated or truncated:
                    break
            
            i_episode += 1
            
            # Log
            writer.add_scalar('Reward/Episode', current_ep_reward, i_episode)
            if train_metrics:
                writer.add_scalar('Loss/Total', train_metrics.get('loss', 0), i_episode)
                writer.add_scalar('Loss/Actor', train_metrics.get('actor_loss', 0), i_episode)
                writer.add_scalar('Loss/Critic', train_metrics.get('critic_loss', 0), i_episode)
                writer.add_scalar('Alpha', train_metrics.get('alpha', 0), i_episode)
            
            print(f"✅ Episode {i_episode} | Steps: {time_step} | Reward: {current_ep_reward:.2f}")
            if train_metrics:
                 print(f"   📈 Loss: {train_metrics.get('loss', 0):.4f}")

        # Final Save
        agent.save(f"checkpoints/{args.exp_name}_final.pth")
        print("[*] Training Finished.")

if __name__ == "__main__":
    main()

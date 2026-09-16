#!/usr/bin/env python3
"""
Jump Guy RL Agent — DQN trained on local physics sim, deployed to Firefox.

Usage:
  python train.py                    # Train on local simulator (fast)
  python train.py --deploy           # Deploy trained agent to Firefox
  python train.py --episodes 500     # Train for 500 episodes
  python train.py --load best.pt     # Load checkpoint
"""

import argparse
import time
import json
import os
from datetime import datetime

from agent import DQNAgent


def train(args):
    from sim import JumpGuySim

    env = JumpGuySim()
    agent = DQNAgent(
        state_dim=8,
        action_dim=2,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
    )

    if args.load and os.path.exists(args.load):
        agent.load(args.load)

    stats = {"scores": [], "epsilons": [], "losses": []}
    best_score = 0

    print(f"\n{'='*60}")
    print(f"  Jump Guy RL — Training (Local Simulator)")
    print(f"  Episodes: {args.episodes}  |  Epsilon: {args.epsilon_start} -> {args.epsilon_end}")
    print(f"{'='*60}\n")

    try:
        for ep in range(1, args.episodes + 1):
            state, info = env.reset()
            total_reward = 0
            score = 0
            steps = 0
            losses = []
            start_time = time.time()

            while True:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()
                if loss > 0:
                    losses.append(loss)

                state = next_state
                total_reward += reward
                score = info.get("score", 0)
                steps += 1

                if done or steps >= args.max_steps:
                    break

            agent.decay_epsilon()
            elapsed = time.time() - start_time
            avg_loss = sum(losses) / len(losses) if losses else 0

            stats["scores"].append(score)
            stats["epsilons"].append(agent.epsilon)
            stats["losses"].append(avg_loss)

            if score > best_score:
                best_score = score
                agent.save(os.path.join(args.save_dir, "best.pt"))

            if ep % args.log_every == 0:
                recent = stats["scores"][-args.log_every:]
                avg = sum(recent) / len(recent)
                print(
                    f"Ep {ep:4d}/{args.episodes} | "
                    f"Score: {score:3d} | Best: {best_score:3d} | "
                    f"Avg({args.log_every}): {avg:5.1f} | "
                    f"Eps: {agent.epsilon:.3f} | "
                    f"Steps: {steps:4d} | "
                    f"{elapsed:.1f}s"
                )

            if ep % args.save_every == 0:
                agent.save(os.path.join(args.save_dir, f"checkpoint_ep{ep}.pt"))
                with open(os.path.join(args.save_dir, "stats.json"), "w") as f:
                    json.dump(stats, f, indent=2)

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted")
    finally:
        agent.save(os.path.join(args.save_dir, "final.pt"))
        with open(os.path.join(args.save_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\n[TRAIN] Best score: {best_score}")
    return stats


def deploy(args):
    """Deploy trained agent to play in real Firefox browser."""
    from env import JumpGuyEnv

    env = JumpGuyEnv(headless=False)
    agent = DQNAgent(state_dim=8, action_dim=2)

    if args.load and os.path.exists(args.load):
        agent.load(args.load)
    else:
        print("[DEPLOY] No checkpoint found, using random policy")

    agent.epsilon = 0.0  # No exploration

    print(f"\n{'='*60}")
    print(f"  Jump Guy RL — Deploying to Firefox")
    print(f"  Watch the Firefox window!")
    print(f"{'='*60}\n")

    try:
        for ep in range(1, args.episodes + 1):
            state, info = env.reset()
            score = 0
            steps = 0

            while True:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                score = info.get("score", 0)
                steps += 1

                if terminated or truncated or steps >= args.max_steps:
                    break

            print(f"  Episode {ep}: Score = {score}, Steps = {steps}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[DEPLOY] Interrupted")
    finally:
        env.close()


def demo_random(args):
    """Demo: random agent playing in Firefox."""
    from env import JumpGuyEnv

    env = JumpGuyEnv(headless=False)
    print(f"\n{'='*60}")
    print(f"  Jump Guy RL — Random Agent Demo in Firefox")
    print(f"{'='*60}\n")

    try:
        for ep in range(1, args.episodes + 1):
            state, info = env.reset()
            score = 0
            steps = 0

            while True:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                score = info.get("score", 0)
                steps += 1

                if terminated or truncated or steps >= args.max_steps:
                    break

            print(f"  Episode {ep}: Score = {score}, Steps = {steps}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[DEMO] Interrupted")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Jump Guy RL Agent")
    parser.add_argument("--deploy", action="store_true", help="Deploy to Firefox")
    parser.add_argument("--demo", action="store_true", help="Random agent demo in Firefox")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.demo:
        demo_random(args)
    elif args.deploy:
        deploy(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

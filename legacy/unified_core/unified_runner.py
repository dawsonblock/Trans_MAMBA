#!/usr/bin/env python3
"""
Unified Runner for Trans_MAMBA + Infinity Research Framework.

Supports:
- LM mode: Synthetic sequence tasks (copy_memory, assoc_recall, etc.)
- RL mode: Reinforcement learning (cartpole, delayed_cue)

Examples:
    python unified_runner.py --mode lm --task copy_memory \
        --controller mamba_dualmem
    python unified_runner.py --mode rl --agent infinity --env cartpole
    python unified_runner.py --mode rl --agent ot --env delayed_cue
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from controllers import (
    TransformerController, TransformerConfig,
    MambaController, MambaConfig,
    MambaDualMemController, MambaDualMemConfig,
)
from synthetic import (
    LMBenchmark, LMBenchConfig,
    SyntheticTaskConfig,
)
from rl import (
    InfinityAgent, InfinityAgentConfig,
    OTMemoryAgent, OTAgentConfig,
    PPOTrainer, PPOConfig,
    make_env,
)

# Optional monitoring imports
try:
    from monitoring import (
        create_detector,
        MonitoredTrainer,
        MonitoredPPOTrainer,
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_lm_controller(name: str, task_cfg: SyntheticTaskConfig):
    """Build LM controller by name."""
    if name == "transformer":
        cfg = TransformerConfig(
            vocab_size=task_cfg.vocab_size,
            d_model=128,
            n_layers=4,
            n_heads=4,
            max_seq_len=task_cfg.seq_len,
        )
        return TransformerController(cfg)

    elif name == "mamba":
        cfg = MambaConfig(
            vocab_size=task_cfg.vocab_size,
            d_model=128,
            n_layers=4,
        )
        return MambaController(cfg)

    elif name == "mamba_dualmem":
        cfg = MambaDualMemConfig(
            vocab_size=task_cfg.vocab_size,
            d_model=128,
            n_layers=4,
            mem_slots=64,
        )
        return MambaDualMemController(cfg)

    else:
        raise ValueError(f"Unknown controller: {name}")


def run_lm_mode(args):
    """Run LM benchmark."""
    print("=" * 60)
    print("LM Mode: Synthetic Sequence Task")
    print("=" * 60)

    set_seed(args.seed)
    device = get_device()

    task_cfg = SyntheticTaskConfig(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_samples=args.num_samples,
        delay=args.delay,
        copy_length=args.copy_length,
        num_pairs=args.num_pairs,
    )

    model = build_lm_controller(args.controller, task_cfg)

    bench_cfg = LMBenchConfig(
        task=args.task,
        controller=args.controller,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        device=str(device),
        seed=args.seed,
    )

    benchmark = LMBenchmark(model, bench_cfg, task_cfg)

    # Wrap with monitoring if enabled
    if args.enable_anomaly_detection and MONITORING_AVAILABLE:
        print("\n[Anomaly Detection ENABLED]")
        detector = create_detector(
            config_path=args.anomaly_config,
            enabled=True,
            verbose=args.verbose_anomalies,
            log_file=args.anomaly_log,
        )
        benchmark = MonitoredTrainer(benchmark, detector, model)
    elif args.enable_anomaly_detection and not MONITORING_AVAILABLE:
        print("\n[WARNING] Anomaly detection requested but not available")

    results = benchmark.train()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.task}_{args.controller}_{timestamp}.json"
        filepath = os.path.join(args.save_dir, filename)

        with open(filepath, "w") as f:
            json.dump({
                "task": args.task,
                "controller": args.controller,
                "config": vars(args),
                "results": {
                    "final_accuracy": results["final_accuracy"],
                    "best_accuracy": results["best_accuracy"],
                },
            }, f, indent=2)

        print(f"\nResults saved to: {filepath}")

    return results


def run_rl_mode(args):
    """Run RL training."""
    print("=" * 60)
    print("RL Mode: Reinforcement Learning")
    print("=" * 60)

    set_seed(args.seed)
    device = get_device()

    env = make_env(
        args.env,
        num_envs=args.num_envs,
        horizon=args.horizon,
        num_actions=args.num_actions,
    )

    if args.agent == "infinity":
        agent_cfg = InfinityAgentConfig(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            d_model=args.d_model,
            n_layers=args.n_layers,
            mem_slots=args.mem_slots,
        )
        agent = InfinityAgent(agent_cfg).to(device)

    elif args.agent == "ot":
        agent_cfg = OTAgentConfig(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            d_model=args.d_model,
            n_layers=args.n_layers,
        )
        agent = OTMemoryAgent(agent_cfg).to(device)

    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    ppo_cfg = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        learning_rate=args.lr,
        ppo_epochs=args.ppo_epochs,
    )

    trainer = PPOTrainer(agent, ppo_cfg, device)

    # Wrap with monitoring if enabled
    if args.enable_anomaly_detection and MONITORING_AVAILABLE:
        print("\n[Anomaly Detection ENABLED]")
        detector = create_detector(
            config_path=args.anomaly_config,
            enabled=True,
            verbose=args.verbose_anomalies,
            log_file=args.anomaly_log,
        )
        trainer = MonitoredPPOTrainer(trainer, detector)
    elif args.enable_anomaly_detection and not MONITORING_AVAILABLE:
        print("\n[WARNING] Anomaly detection requested but not available")

    print(f"\nAgent: {args.agent}")
    print(f"Environment: {args.env}")
    print(f"Device: {device}")
    print(f"Total updates: {args.num_updates}")
    print()

    total_reward = 0.0
    best_reward = float("-inf")
    start_time = time.time()

    for update in range(1, args.num_updates + 1):
        rollout_reward, next_value = trainer.collect_rollout(
            env, args.rollout_length
        )
        metrics = trainer.update(next_value)

        total_reward += rollout_reward

        if update % args.log_interval == 0:
            avg_reward = total_reward / args.log_interval
            elapsed = time.time() - start_time

            print(
                f"Update {update}/{args.num_updates} | "
                f"Reward: {avg_reward:.2f} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if avg_reward > best_reward:
                best_reward = avg_reward

            total_reward = 0.0

    env.close()

    print(f"\nTraining complete. Best reward: {best_reward:.2f}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.env}_{args.agent}_{timestamp}.json"
        filepath = os.path.join(args.save_dir, filename)

        with open(filepath, "w") as f:
            json.dump({
                "env": args.env,
                "agent": args.agent,
                "config": vars(args),
                "results": {"best_reward": best_reward},
            }, f, indent=2)

        print(f"Results saved to: {filepath}")

    return {"best_reward": best_reward}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Runner for Trans_MAMBA + Infinity"
    )

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["lm", "rl"],
        help="Mode: lm (language model) or rl (reinforcement learning)"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)

    lm_group = parser.add_argument_group("LM Mode")
    lm_group.add_argument(
        "--task", type=str, default="copy_memory",
        choices=[
            "copy_memory",
            "assoc_recall",
            "selective_copy",
            "induction_head",
        ]
    )
    lm_group.add_argument(
        "--controller", type=str, default="transformer",
        choices=["transformer", "mamba", "mamba_dualmem"]
    )
    lm_group.add_argument("--seq_len", type=int, default=128)
    lm_group.add_argument("--vocab_size", type=int, default=16)
    lm_group.add_argument("--num_samples", type=int, default=10000)
    lm_group.add_argument("--delay", type=int, default=40)
    lm_group.add_argument("--copy_length", type=int, default=10)
    lm_group.add_argument("--num_pairs", type=int, default=4)
    lm_group.add_argument("--epochs", type=int, default=20)
    lm_group.add_argument("--batch_size", type=int, default=32)

    rl_group = parser.add_argument_group("RL Mode")
    rl_group.add_argument(
        "--agent", type=str, default="infinity",
        choices=["infinity", "ot"]
    )
    rl_group.add_argument(
        "--env", type=str, default="cartpole",
        choices=["cartpole", "delayed_cue"]
    )
    rl_group.add_argument("--num_envs", type=int, default=8)
    rl_group.add_argument("--num_updates", type=int, default=100)
    rl_group.add_argument("--rollout_length", type=int, default=128)
    rl_group.add_argument("--horizon", type=int, default=40)
    rl_group.add_argument("--num_actions", type=int, default=4)
    rl_group.add_argument("--d_model", type=int, default=64)
    rl_group.add_argument("--n_layers", type=int, default=2)
    rl_group.add_argument("--mem_slots", type=int, default=32)
    rl_group.add_argument("--gamma", type=float, default=0.99)
    rl_group.add_argument("--gae_lambda", type=float, default=0.95)
    rl_group.add_argument("--clip_eps", type=float, default=0.2)
    rl_group.add_argument("--ppo_epochs", type=int, default=4)
    rl_group.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--lr", type=float, default=3e-4)

    # Anomaly detection arguments
    monitor_group = parser.add_argument_group("Anomaly Detection")
    monitor_group.add_argument(
        "--enable_anomaly_detection", action="store_true",
        help="Enable runtime anomaly detection"
    )
    monitor_group.add_argument(
        "--anomaly_config", type=str, default=None,
        help="Path to anomaly detection config JSON"
    )
    monitor_group.add_argument(
        "--anomaly_log", type=str, default=None,
        help="Path to anomaly log file"
    )
    monitor_group.add_argument(
        "--verbose_anomalies", action="store_true",
        help="Print anomalies when detected"
    )

    args = parser.parse_args()

    if args.mode == "lm":
        run_lm_mode(args)
    else:
        run_rl_mode(args)


if __name__ == "__main__":
    main()

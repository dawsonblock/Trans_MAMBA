#!/usr/bin/env python3
"""
Unified Runner for Lean Infinity Dual Hybrid v1.

Supports:
- LM mode: Synthetic sequence tasks (copy_memory, assoc_recall, etc.)
- RL mode: Reinforcement learning (cartpole, delayed_cue)

Examples:
    python -m trans_mamba_core.unified_runner --mode lm --task copy_memory \
        --controller mamba_dualmem
    python -m trans_mamba_core.unified_runner --mode rl --agent infinity \
        --env cartpole
    python -m trans_mamba_core.unified_runner --mode rl --agent ot \
        --env delayed_cue
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch

from trans_mamba_core.controllers import (
    MambaConfig,
    MambaController,
    MambaDualMemConfig,
    MambaDualMemController,
    TransformerConfig,
    TransformerController,
)
from trans_mamba_core.rl import (
    InfinityAgent,
    InfinityAgentConfig,
    OTAgentConfig,
    OTMemoryAgent,
    PPOConfig,
    PPOTrainer,
    make_env,
)
from trans_mamba_core.synthetic import (
    LMBenchConfig,
    LMBenchmark,
    SyntheticTaskConfig,
)

# Optional monitoring imports (available after monitoring migration)
try:
    from trans_mamba_core.monitoring import (
        MonitoredPPOTrainer,
        MonitoredTrainer,
        create_detector,
    )

    MONITORING_AVAILABLE = True
except Exception:
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


def build_lm_controller(
    name: str,
    task_cfg: SyntheticTaskConfig,
    d_model: int,
    n_layers: int,
    mem_slots: int,
):
    """Build LM controller by name."""

    if name == "transformer":
        cfg = TransformerConfig(
            vocab_size=task_cfg.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=4,
            max_seq_len=task_cfg.seq_len,
        )
        return TransformerController(cfg)

    if name == "mamba":
        cfg = MambaConfig(
            vocab_size=task_cfg.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
        )
        return MambaController(cfg)

    if name == "mamba_dualmem":
        cfg = MambaDualMemConfig(
            vocab_size=task_cfg.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            mem_slots=mem_slots,
        )
        return MambaDualMemController(cfg)

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

    model = build_lm_controller(
        args.controller,
        task_cfg,
        d_model=args.d_model,
        n_layers=args.n_layers,
        mem_slots=args.mem_slots,
    )

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

    out_dir = args.out_dir or args.save_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    results = benchmark.train()

    if out_dir:
        eval_metrics = benchmark.evaluate()
        metrics_path = os.path.join(out_dir, "metrics.jsonl")
        with open(metrics_path, "w") as f:
            history = getattr(benchmark, "metrics_history", [])
            for m in history:
                f.write(
                    json.dumps(
                        {
                            "epoch": m.get("epoch"),
                            "train_loss": m.get("loss"),
                            "eval_loss": eval_metrics.get("eval_loss"),
                            "accuracy": m.get("accuracy"),
                        }
                    )
                    + "\n"
                )
        torch.save(model.state_dict(), os.path.join(out_dir, "final.pt"))

    if args.save_dir and not args.out_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.task}_{args.controller}_{timestamp}.json"
        filepath = os.path.join(args.save_dir, filename)

        with open(filepath, "w") as f:
            json.dump(
                {
                    "task": args.task,
                    "controller": args.controller,
                    "config": vars(args),
                    "results": {
                        "final_accuracy": results["final_accuracy"],
                        "best_accuracy": results["best_accuracy"],
                    },
                },
                f,
                indent=2,
            )

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

    out_dir = args.out_dir or args.save_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        metrics_f = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    else:
        metrics_f = None

    total_reward = 0.0
    best_reward = float("-inf")
    start_time = time.time()

    for update in range(1, args.num_updates + 1):
        rollout_reward, next_value = trainer.collect_rollout(
            env,
            args.rollout_length,
        )
        metrics = trainer.update(next_value)

        total_reward += rollout_reward

        if metrics_f is not None:
            mean_return = float(rollout_reward) / max(
                float(args.num_envs),
                1.0,
            )
            metrics_f.write(
                json.dumps(
                    {
                        "update": update,
                        "mean_return": mean_return,
                        "policy_loss": float(metrics.get("policy_loss", 0.0)),
                        "value_loss": float(metrics.get("value_loss", 0.0)),
                        "entropy": float(metrics.get("entropy", 0.0)),
                        "kl": float(metrics.get("approx_kl", 0.0)),
                        "clipfrac": float(metrics.get("clipfrac", 0.0)),
                    }
                )
                + "\n"
            )

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

    if metrics_f is not None:
        metrics_f.close()

    print(f"\nTraining complete. Best reward: {best_reward:.2f}")

    if out_dir:
        torch.save(agent.state_dict(), os.path.join(out_dir, "final.pt"))

    if args.save_dir and not args.out_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{args.env}_{args.agent}_{timestamp}.json"
        filepath = os.path.join(args.save_dir, filename)

        with open(filepath, "w") as f:
            json.dump(
                {
                    "env": args.env,
                    "agent": args.agent,
                    "config": vars(args),
                    "results": {"best_reward": best_reward},
                },
                f,
                indent=2,
            )

        print(f"Results saved to: {filepath}")

    return {"best_reward": best_reward}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Runner for Lean Infinity Dual Hybrid v1"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["lm", "rl"],
        help="Mode: lm (language model) or rl (reinforcement learning)",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for run artifacts (alias of save_dir)",
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Model width for LM/RL controllers and agents",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of layers for LM/RL controllers and agents",
    )
    parser.add_argument(
        "--mem_slots",
        type=int,
        default=64,
        help="Memory slots for DualMem controllers/agents",
    )

    lm_group = parser.add_argument_group("LM Mode")
    lm_group.add_argument(
        "--task",
        type=str,
        default="copy_memory",
        choices=[
            "copy_memory",
            "assoc_recall",
            "selective_copy",
            "induction_head",
        ],
    )
    lm_group.add_argument(
        "--controller",
        type=str,
        default="transformer",
        choices=["transformer", "mamba", "mamba_dualmem"],
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
        "--agent",
        type=str,
        default="infinity",
        choices=["infinity", "ot"],
    )
    rl_group.add_argument(
        "--env",
        type=str,
        default="cartpole",
        choices=["cartpole", "delayed_cue"],
    )
    rl_group.add_argument("--num_envs", type=int, default=8)
    rl_group.add_argument("--num_updates", type=int, default=100)
    rl_group.add_argument("--rollout_length", type=int, default=128)
    rl_group.add_argument("--horizon", type=int, default=40)
    rl_group.add_argument("--num_actions", type=int, default=4)
    rl_group.add_argument("--gamma", type=float, default=0.99)
    rl_group.add_argument("--gae_lambda", type=float, default=0.95)
    rl_group.add_argument("--clip_eps", type=float, default=0.2)
    rl_group.add_argument("--ppo_epochs", type=int, default=4)
    rl_group.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--lr", type=float, default=3e-4)

    monitor_group = parser.add_argument_group("Anomaly Detection")
    monitor_group.add_argument(
        "--enable_anomaly_detection",
        action="store_true",
        help="Enable runtime anomaly detection",
    )
    monitor_group.add_argument(
        "--anomaly_config",
        type=str,
        default=None,
        help="Path to anomaly detection config JSON",
    )
    monitor_group.add_argument(
        "--anomaly_log",
        type=str,
        default=None,
        help="Path to anomaly log file",
    )
    monitor_group.add_argument(
        "--verbose_anomalies",
        action="store_true",
        help="Print anomalies when detected",
    )

    args = parser.parse_args()

    if args.mode == "lm":
        run_lm_mode(args)
    else:
        run_rl_mode(args)


if __name__ == "__main__":
    main()

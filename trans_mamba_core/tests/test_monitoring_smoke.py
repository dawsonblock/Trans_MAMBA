import torch


def test_monitoring_lm_and_ppo_smoke():
    from trans_mamba_core.controllers import (
        MambaConfig,
        MambaController,
    )
    from trans_mamba_core.monitoring import (
        MonitoredPPOTrainer,
        MonitoredTrainer,
        create_detector,
    )
    from trans_mamba_core.rl import PPOConfig, PPOTrainer, make_env
    from trans_mamba_core.synthetic import (
        LMBenchConfig,
        LMBenchmark,
        SyntheticTaskConfig,
    )

    device = torch.device("cpu")

    detector = create_detector(enabled=True)

    task_cfg = SyntheticTaskConfig(
        seq_len=16,
        vocab_size=16,
        num_samples=128,
        delay=4,
        copy_length=4,
        num_pairs=2,
    )
    model = MambaController(
        MambaConfig(vocab_size=16, d_model=32, n_layers=1)
    )
    bench_cfg = LMBenchConfig(
        task="copy_memory",
        controller="mamba",
        batch_size=8,
        learning_rate=1e-3,
        epochs=1,
        device=str(device),
        seed=0,
    )
    bench = LMBenchmark(model, bench_cfg, task_cfg)
    monitored_bench = MonitoredTrainer(bench, detector, model)

    batch = next(iter(bench.train_loader))
    metrics = monitored_bench.train_step(batch)
    assert "loss" in metrics

    detector2 = create_detector(enabled=True)
    env = make_env("delayed_cue", num_envs=2, horizon=8, num_actions=2)

    from trans_mamba_core.rl import InfinityAgent, InfinityAgentConfig

    agent_cfg = InfinityAgentConfig(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        d_model=32,
        n_layers=1,
        mem_slots=8,
    )
    agent = InfinityAgent(agent_cfg).to(device)

    ppo_cfg = PPOConfig(
        ppo_epochs=1,
        num_minibatches=1,
        learning_rate=1e-3,
    )
    trainer = PPOTrainer(agent, ppo_cfg, device)
    monitored_trainer = MonitoredPPOTrainer(trainer, detector2)

    reward, next_value = monitored_trainer.collect_rollout(
        env,
        rollout_length=4,
    )
    assert isinstance(reward, float)

    out = monitored_trainer.update(next_value)
    assert isinstance(out, dict)

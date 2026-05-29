from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.btc_direction_learning.algorithms import behavior_cloning, dagger, ppo
from src.btc_direction_learning.config import (
    ARTIFACT_ROOT,
    DEFAULT_SPLIT_REGIME,
    DEFAULT_TEST_HOURS,
    DEFAULT_TRAIN_HOURS,
    FIXED_REGIME_NAME,
    FIXED_TEST_POOL_HOURS,
    FIXED_TRAIN_POOL_HOURS,
    SHARED_DATA_DIR,
    build_env_slug,
    build_regime_split_slug,
    get_algorithm_output_dir,
)
from src.btc_direction_learning.dataset import (
    DirectionDatasetBundle,
    build_direction_dataset_bundle,
    build_direction_dataset_bundle_from_processed,
    ensure_shared_data_dir,
)
from src.btc_direction_learning.env import (
    BTCDirectionEnv,
    ENV_VERSION_BINARY,
    ENV_VERSION_INTENSITY11,
    ENV_VERSION_TERNARY,
    baseline_down_action,
    baseline_up_action,
)
from src.btc_direction_learning.evaluation import evaluate_policy, load_checkpoint, simulate_portfolio
from src.btc_direction_learning.plotting import create_portfolio_simulation_plot, create_training_summary_plot
from src.utils.market_data import set_seed


ALGORITHMS = {
    "ppo": ppo,
    "bc": behavior_cloning,
    "dagger": dagger,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shared BTC direction-learning model.")
    parser.add_argument("--algo", choices=sorted(ALGORITHMS.keys()), required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-hours", type=int, default=DEFAULT_TRAIN_HOURS)
    parser.add_argument("--test-hours", type=int, default=DEFAULT_TEST_HOURS)
    parser.add_argument("--raw-fetch-hours", type=int, default=0)
    parser.add_argument("--split-regime", choices=["variable", "fixed_5k_5k"], default=DEFAULT_SPLIT_REGIME)
    parser.add_argument("--data-variant", choices=["full", "market_hours"], default="full")
    parser.add_argument(
        "--env-version",
        choices=[ENV_VERSION_BINARY, ENV_VERSION_TERNARY, ENV_VERSION_INTENSITY11],
        default=ENV_VERSION_BINARY,
    )
    parser.add_argument("--ternary-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--ppo-total-updates", type=int, default=0)
    parser.add_argument("--ppo-trajectories-per-update", type=int, default=0)
    parser.add_argument("--ppo-epochs", type=int, default=0)
    parser.add_argument("--ppo-minibatch-size", type=int, default=0)
    parser.add_argument("--ppo-learning-rate", type=float, default=0.0)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.0)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.0)
    parser.add_argument("--continual-lr-decay", type=float, default=1.0)
    parser.add_argument("--continual-clip-decay", type=float, default=1.0)
    parser.add_argument("--continual-entropy-decay", type=float, default=1.0)
    parser.add_argument("--previous-policy-kl-coef", type=float, default=0.0)
    parser.add_argument("--early-stopping-enabled", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stopping-restore-best", action="store_true")
    return parser.parse_args()


def build_fixed_regime_window_bundle(
    canonical_bundle: DirectionDatasetBundle,
    train_hours: int,
    test_hours: int,
) -> DirectionDatasetBundle:
    if train_hours > FIXED_TRAIN_POOL_HOURS:
        raise RuntimeError(f"--train-hours cannot exceed {FIXED_TRAIN_POOL_HOURS} in fixed_5k_5k regime.")
    if test_hours > FIXED_TEST_POOL_HOURS:
        raise RuntimeError(f"--test-hours cannot exceed {FIXED_TEST_POOL_HOURS} in fixed_5k_5k regime.")
    train_start = FIXED_TRAIN_POOL_HOURS - train_hours
    test_end = FIXED_TRAIN_POOL_HOURS + test_hours
    processed = canonical_bundle.processed_df.iloc[train_start:test_end].reset_index(drop=True)
    return build_direction_dataset_bundle_from_processed(
        processed=processed,
        train_hours=train_hours,
        test_hours=test_hours,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(output_dir: Path, checkpoint_name: str, state_dict: dict[str, torch.Tensor]) -> Path:
    checkpoint_path = output_dir / f"policy_{checkpoint_name}.pt"
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def build_portfolio_scenarios(
    actions: list[int],
    labels: list[int],
    env_version: str,
) -> dict[str, dict]:
    return {
        "fixed_dollar": simulate_portfolio(
            actions=actions,
            labels=labels,
            env_version=env_version,
            starting_balance=10.0,
            move_fraction=0.05,
            sizing_mode="fixed_dollar",
        ),
        "current_fraction": simulate_portfolio(
            actions=actions,
            labels=labels,
            env_version=env_version,
            starting_balance=10.0,
            move_fraction=0.05,
            sizing_mode="current_fraction",
        ),
        "peak_fraction": simulate_portfolio(
            actions=actions,
            labels=labels,
            env_version=env_version,
            starting_balance=10.0,
            move_fraction=0.05,
            sizing_mode="peak_fraction",
        ),
    }


def run_training(args: argparse.Namespace) -> dict:
    dataset_bundle = build_direction_dataset_bundle(
        force_refresh=args.force_refresh,
        train_hours=args.train_hours,
        test_hours=args.test_hours,
        raw_fetch_hours=args.raw_fetch_hours or None,
        split_regime=args.split_regime,
        data_variant=args.data_variant,
        data_dir=ensure_shared_data_dir(SHARED_DATA_DIR),
    )
    if args.split_regime == FIXED_REGIME_NAME:
        dataset_bundle = build_fixed_regime_window_bundle(dataset_bundle, args.train_hours, args.test_hours)
    return run_training_with_bundle(args, dataset_bundle)


def run_training_with_bundle(
    args: argparse.Namespace,
    dataset_bundle: DirectionDatasetBundle,
    output_dir_override: Path | None = None,
    split_slug_override: str | None = None,
    shared_data_dir_override: Path | None = None,
    initial_policy_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict:
    algorithm_name = args.algo
    if args.env_version == ENV_VERSION_INTENSITY11 and algorithm_name != "ppo":
        raise RuntimeError("env_version=intensity11 is only supported with algo=ppo.")
    split_slug = split_slug_override or build_regime_split_slug(args.split_regime, args.train_hours, args.test_hours)
    env_slug = build_env_slug(args.env_version)
    resolved_output_dir = (
        output_dir_override
        if output_dir_override is not None
        else (
            Path(args.output_dir)
            if args.output_dir
            else (ARTIFACT_ROOT / env_slug / split_slug / algorithm_name)
        )
    )
    output_dir = get_algorithm_output_dir(algorithm_name, output_dir=resolved_output_dir)
    shared_data_dir = shared_data_dir_override or ensure_shared_data_dir(SHARED_DATA_DIR)

    set_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    train_env = BTCDirectionEnv(dataset_bundle=dataset_bundle, split_name="train", env_version=args.env_version)
    test_env = BTCDirectionEnv(dataset_bundle=dataset_bundle, split_name="test", env_version=args.env_version)

    algorithm = ALGORITHMS[algorithm_name]
    policy = algorithm.build_policy(
        sequence_length=dataset_bundle.sequence_length,
        feature_dim=dataset_bundle.feature_dim,
        action_dim=train_env.action_dim,
    ).to(device)
    if initial_policy_state_dict is not None:
        policy.load_state_dict(initial_policy_state_dict)
    train_target = dataset_bundle if algorithm_name in {"bc", "dagger"} else train_env
    train_kwargs: dict[str, object] = {}
    if algorithm_name == "ppo":
        is_continual_run = initial_policy_state_dict is not None
        if args.ppo_total_updates > 0:
            train_kwargs["total_updates"] = args.ppo_total_updates
        if args.ppo_trajectories_per_update > 0:
            train_kwargs["trajectories_per_update"] = args.ppo_trajectories_per_update
        if args.ppo_epochs > 0:
            train_kwargs["ppo_epochs"] = args.ppo_epochs
        if args.ppo_minibatch_size > 0:
            train_kwargs["minibatch_size"] = args.ppo_minibatch_size
        if args.ppo_learning_rate > 0:
            learning_rate = float(args.ppo_learning_rate)
            if is_continual_run:
                learning_rate *= float(args.continual_lr_decay)
            train_kwargs["learning_rate"] = learning_rate
        if args.ppo_clip_epsilon > 0:
            clip_epsilon = float(args.ppo_clip_epsilon)
            if is_continual_run:
                clip_epsilon *= float(args.continual_clip_decay)
            train_kwargs["clip_epsilon"] = clip_epsilon
        if args.ppo_entropy_coef > 0:
            entropy_coef = float(args.ppo_entropy_coef)
            if is_continual_run:
                entropy_coef *= float(args.continual_entropy_decay)
            train_kwargs["entropy_coef"] = entropy_coef
        if args.previous_policy_kl_coef > 0 and is_continual_run:
            train_kwargs["previous_policy_kl_coef"] = float(args.previous_policy_kl_coef)
            train_kwargs["reference_policy_state_dict"] = initial_policy_state_dict
        if args.early_stopping_enabled:
            train_kwargs["early_stopping_patience"] = int(args.early_stopping_patience)
            train_kwargs["early_stopping_min_delta"] = float(args.early_stopping_min_delta)
            train_kwargs["early_stopping_restore_best"] = bool(args.early_stopping_restore_best)
    if algorithm_name == "dagger":
        train_kwargs["env_version"] = args.env_version
        train_kwargs["ternary_confidence_threshold"] = args.ternary_confidence_threshold

    history, checkpoints, plot_config = algorithm.train(
        train_target,
        policy,
        device,
        **train_kwargs,
    )

    evaluations: dict[str, dict[str, dict]] = {}
    checkpoint_paths: dict[str, str] = {}
    for checkpoint_name, state_dict in checkpoints.items():
        checkpoint_paths[checkpoint_name] = str(save_checkpoint(output_dir, checkpoint_name, state_dict))
        load_checkpoint(policy, state_dict)
        evaluations[checkpoint_name] = {
            "train": evaluate_policy(
                train_env,
                policy,
                device=device,
                ternary_confidence_threshold=args.ternary_confidence_threshold,
            ),
            "test": evaluate_policy(
                test_env,
                policy,
                device=device,
                ternary_confidence_threshold=args.ternary_confidence_threshold,
            ),
        }

    best_train_checkpoint_name = max(
        evaluations,
        key=lambda checkpoint_name: float(evaluations[checkpoint_name]["train"]["total_reward"]),
    )
    checkpoints["best_train"] = checkpoints[best_train_checkpoint_name]
    checkpoint_paths["best_train"] = str(save_checkpoint(output_dir, "best_train", checkpoints["best_train"]))
    evaluations["best_train"] = {
        "train": dict(evaluations[best_train_checkpoint_name]["train"]),
        "test": dict(evaluations[best_train_checkpoint_name]["test"]),
    }

    history_path = output_dir / "training_metrics.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    processed_path = output_dir / "processed_dataset.csv"
    dataset_bundle.processed_df.to_csv(processed_path, index=False)

    plot_path = output_dir / "training_summary.png"
    create_training_summary_plot(
        history=history,
        evaluations=evaluations,
        output_path=plot_path,
        learning_curve_title=plot_config["title"],
        learning_curve_series=plot_config["series"],
    )

    random_generator = np.random.default_rng(args.seed)
    random_actions = random_generator.integers(
        low=0,
        high=train_env.action_dim,
        size=len(evaluations["best_train"]["test"]["labels"]),
        dtype=np.int64,
    ).tolist()
    full_up_actions = [baseline_up_action(args.env_version) for _ in evaluations["best_train"]["test"]["labels"]]
    full_down_actions = [baseline_down_action(args.env_version) for _ in evaluations["best_train"]["test"]["labels"]]
    portfolio_groups = {
        "random": build_portfolio_scenarios(
            actions=random_actions,
            labels=evaluations["best_train"]["test"]["labels"],
            env_version=args.env_version,
        ),
        "full_up": build_portfolio_scenarios(
            actions=full_up_actions,
            labels=evaluations["best_train"]["test"]["labels"],
            env_version=args.env_version,
        ),
        "full_down": build_portfolio_scenarios(
            actions=full_down_actions,
            labels=evaluations["best_train"]["test"]["labels"],
            env_version=args.env_version,
        ),
        "initial": build_portfolio_scenarios(
            actions=evaluations["initial"]["test"]["actions"],
            labels=evaluations["initial"]["test"]["labels"],
            env_version=args.env_version,
        ),
        "best_train": build_portfolio_scenarios(
            actions=evaluations["best_train"]["test"]["actions"],
            labels=evaluations["best_train"]["test"]["labels"],
            env_version=args.env_version,
        ),
    }
    portfolio_plot_path = output_dir / "best_train_test_portfolio.png"
    create_portfolio_simulation_plot(
        portfolio_groups=portfolio_groups,
        output_path=portfolio_plot_path,
        title="Best Train Model Test Portfolio Simulation",
    )

    summary = {
        "algorithm": algorithm_name,
        "device": str(device),
        "env_version": args.env_version,
        "split_regime": args.split_regime,
        "data_variant": args.data_variant,
        "ternary_confidence_threshold": args.ternary_confidence_threshold,
        "artifact_root": str(ARTIFACT_ROOT),
        "shared_data_dir": str(shared_data_dir / split_slug),
        "split_slug": split_slug,
        "history_path": str(history_path),
        "processed_path": str(processed_path),
        "plot_path": str(plot_path),
        "portfolio_plot_path": str(portfolio_plot_path),
        "checkpoint_paths": checkpoint_paths,
        "best_train_checkpoint_name": best_train_checkpoint_name,
        "initialized_from_previous_window": bool(initial_policy_state_dict is not None),
        "portfolio_groups": portfolio_groups,
        "early_stopping": plot_config.get("early_stopping", {}),
        "train_overrides": {
            key: value
            for key, value in train_kwargs.items()
            if key != "reference_policy_state_dict"
        },
        "evaluations": evaluations,
        **dataset_bundle.get_all_metadata(),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_training(args)
    accuracy_summary = {
        checkpoint_name: {
            split_name: round(float(result["accuracy"]), 4)
            for split_name, result in checkpoint_results.items()
        }
        for checkpoint_name, checkpoint_results in summary["evaluations"].items()
        if checkpoint_name in {"initial", "midpoint", "best_train"}
    }
    print(
        json.dumps(
            {
                "algorithm": summary["algorithm"],
                "split_regime": summary["split_regime"],
                "data_variant": summary["data_variant"],
                "env_version": summary["env_version"],
                "ternary_confidence_threshold": summary["ternary_confidence_threshold"],
                "split_slug": summary["split_slug"],
                "best_train_checkpoint_name": summary["best_train_checkpoint_name"],
                "train_rows": summary["train_rows"],
                "test_rows": summary["test_rows"],
                "train_decisions": summary["train_decisions"],
                "test_decisions": summary["test_decisions"],
                "plot_path": summary["plot_path"],
                "portfolio_plot_path": summary["portfolio_plot_path"],
                "run_summary_path": str(
                    Path(summary["history_path"]).with_name("run_summary.json")
                ),
                "accuracies": accuracy_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

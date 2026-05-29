from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def create_training_summary_plot(
    history: list[dict],
    evaluations: dict[str, dict[str, dict]],
    output_path: Path,
    learning_curve_title: str,
    learning_curve_series: list[dict[str, str]],
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    flat_axes = axes.flatten()

    ax = flat_axes[0]
    x_axis = list(range(1, len(history) + 1))
    if history and "step" in history[0]:
        x_axis = [int(row["step"]) for row in history]
    elif history and "update" in history[0]:
        x_axis = [int(row["update"]) for row in history]
    elif history and "epoch" in history[0]:
        x_axis = [int(row["epoch"]) for row in history]
    elif history and "round" in history[0]:
        x_axis = [int(row["round"]) for row in history]

    for index, series in enumerate(learning_curve_series):
        values = [float(row[series["key"]]) for row in history]
        ax.plot(
            x_axis,
            values,
            label=series["label"],
            linewidth=2,
            color=series.get("color", None),
            alpha=float(series.get("alpha", 1.0)),
        )
    ax.set_title(learning_curve_title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Metric")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.legend()
    ax.grid(alpha=0.25)

    layout = [
        ("initial", "train", "Train Rewards - 0% Model"),
        ("midpoint", "train", "Train Rewards - 50% Model"),
        ("best_train", "train", "Train Rewards - Best Train Model"),
        ("initial", "test", "Test Rewards - 0% Model"),
        ("midpoint", "test", "Test Rewards - 50% Model"),
        ("best_train", "test", "Test Rewards - Best Train Model"),
    ]

    for axis, (checkpoint_name, split_name, title) in zip(flat_axes[1:7], layout):
        result = evaluations[checkpoint_name][split_name]
        cumulative = result["cumulative_rewards"]
        steps = list(range(len(cumulative)))
        axis.plot(steps, cumulative, marker="o", linewidth=2, color="#2ca02c")
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.set_ylabel("Cumulative reward")
        axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.grid(alpha=0.25)
        axis.text(
            0.02,
            0.95,
            f"mean={result['mean_reward']:.2f}\ntotal={result['total_reward']:.0f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )

    accuracy_axis = flat_axes[7]
    checkpoint_order = ["initial", "midpoint", "best_train"]
    checkpoint_labels = ["0%", "50%", "Best Train"]
    train_accuracies = [float(evaluations[name]["train"]["accuracy"]) for name in checkpoint_order]
    test_accuracies = [float(evaluations[name]["test"]["accuracy"]) for name in checkpoint_order]
    x_positions = list(range(len(checkpoint_order)))
    bar_width = 0.35

    accuracy_axis.bar(
        [position - bar_width / 2 for position in x_positions],
        train_accuracies,
        width=bar_width,
        color="#1f77b4",
        label="Train",
    )
    accuracy_axis.bar(
        [position + bar_width / 2 for position in x_positions],
        test_accuracies,
        width=bar_width,
        color="#ff7f0e",
        label="Test",
    )
    accuracy_axis.set_title("Accuracy by Checkpoint")
    accuracy_axis.set_xlabel("Training progress")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_xticks(x_positions, checkpoint_labels)
    accuracy_axis.set_ylim(0.0, 1.0)
    accuracy_axis.axhline(0.5, color="black", linewidth=1, linestyle="--", label="50% threshold")
    accuracy_axis.grid(axis="y", alpha=0.25)
    accuracy_axis.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_portfolio_simulation_plot(
    portfolio_groups: dict[str, dict[str, dict]],
    output_path: Path,
    title: str,
) -> None:
    column_layout = [
        ("fixed_dollar", "Conservative: +/- $1", "#1f77b4"),
        ("current_fraction", "5% of Current Portfolio", "#ff7f0e"),
        ("peak_fraction", "5% of Max Portfolio So Far", "#2ca02c"),
    ]
    row_layout = [
        ("random", "Random Choices"),
        ("full_up", "Always UP"),
        ("full_down", "Always DOWN"),
        ("initial", "Initial Model"),
        ("best_train", "Best Train Model"),
    ]
    fig, axes = plt.subplots(len(row_layout), 3, figsize=(18, 22), sharex=True)
    fig.suptitle(title)

    for row_index, (group_key, row_title) in enumerate(row_layout):
        for col_index, (sizing_key, panel_title, color) in enumerate(column_layout):
            axis = axes[row_index][col_index]
            portfolio = portfolio_groups[group_key][sizing_key]
            balances = [float(value) for value in portfolio["balances"]]
            steps = list(range(len(balances)))
            axis.plot(steps, balances, linewidth=2.5, color=color)
            axis.axhline(
                float(portfolio["starting_balance"]),
                color="black",
                linewidth=1,
                linestyle="--",
                label=f"Start {portfolio['starting_balance']:.0f}",
            )
            peak_idx = int(portfolio.get("max_drawdown_peak_index", 0))
            trough_idx = int(portfolio.get("max_drawdown_trough_index", 0))
            if trough_idx > peak_idx:
                axis.axvline(peak_idx, color="#d62728", linewidth=1.5, linestyle="--", label="Largest fall start")
                axis.axvline(trough_idx, color="#9467bd", linewidth=1.5, linestyle="--", label="Largest fall end")
            axis.set_title(f"{row_title} | {panel_title}")
            axis.set_ylabel("Portfolio value")
            axis.grid(alpha=0.25)
            axis.legend()
            axis.text(
                0.02,
                0.95,
                (
                    f"final={portfolio['final_balance']:.2f}\n"
                    f"pnl={portfolio['pnl']:.2f}\n"
                    f"pnl_pct={portfolio['pnl_pct'] * 100:.1f}%\n"
                    f"max={portfolio['max_balance']:.2f}\n"
                    f"min={portfolio['min_balance']:.2f}\n"
                    f"largest_fall={portfolio['max_drawdown_pct'] * 100:.1f}%\n"
                    f"worst_start_to_end={portfolio['max_drawdown_segment_loss_score']}\n"
                    f"win_streak={portfolio['longest_win_streak']}\n"
                    f"loss_streak={portfolio['longest_loss_streak']}\n"
                    f"full_loss_times={portfolio['full_loss_count']}"
                ),
                transform=axis.transAxes,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
            )

    for axis in axes[-1]:
        axis.set_xlabel("Step")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_merged_test_summary_plot(
    strategy_results: dict[str, dict],
    output_path: Path,
    title: str,
) -> None:
    strategy_layout = [
        ("random", "Random", "#7f7f7f"),
        ("full_up", "Always UP", "#1f77b4"),
        ("full_down", "Always DOWN", "#ff7f0e"),
        ("initial", "Initial", "#9467bd"),
        ("best_train", "Best Train", "#2ca02c"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(title)

    reward_axis = axes[0]
    for strategy_key, label, color in strategy_layout:
        result = strategy_results[strategy_key]
        reward_axis.plot(
            list(range(len(result["cumulative_rewards"]))),
            result["cumulative_rewards"],
            linewidth=2,
            color=color,
            label=f"{label} (total={result['total_reward']:.0f})",
        )
    reward_axis.set_title("Merged Test Cumulative Rewards")
    reward_axis.set_xlabel("Merged test step")
    reward_axis.set_ylabel("Cumulative reward")
    reward_axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    reward_axis.grid(alpha=0.25)
    reward_axis.legend()

    accuracy_axis = axes[1]
    labels = [label for _, label, _ in strategy_layout]
    colors = [color for _, _, color in strategy_layout]
    accuracies = [float(strategy_results[key]["accuracy"]) for key, _, _ in strategy_layout]
    accuracy_axis.bar(labels, accuracies, color=colors)
    accuracy_axis.axhline(0.5, color="black", linewidth=1, linestyle="--", label="50% threshold")
    accuracy_axis.set_title("Merged Test Accuracy")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, 1.0)
    accuracy_axis.grid(axis="y", alpha=0.25)
    accuracy_axis.legend()

    for bar_label, accuracy in zip(labels, accuracies):
        accuracy_axis.text(bar_label, accuracy + 0.02, f"{accuracy * 100:.1f}%", ha="center", va="bottom")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

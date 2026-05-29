"""
Utility to generate markdown tables from experiment summary.json files and
automatically update EXPERIMENTS.md with the results.

Usage:
    python scripts/generate_experiment_tables.py --experiment-num 1 --variant full
    python scripts/generate_experiment_tables.py --summary-path EXPERIMENTS/1/full/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EASTERN_TZ = ZoneInfo("America/New_York")
FAMILY_ORDER = [
    "RANDOM",
    "ALWAYS_UP",
    "ALWAYS_DOWN",
    "BC",
    "DAGGER",
    "NN",
    "RF",
    "XGBOOST",
    "LSTM",
    "TRANSFORMER",
    "LSTMPPO",
    "PPO",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate markdown tables from experiment summaries and update EXPERIMENTS.md"
    )
    parser.add_argument("--experiment-num", type=int, help="Experiment number (1-4, etc.)")
    parser.add_argument("--summary-path", type=str, help="Direct path to summary.json")
    parser.add_argument("--variant", default="full", help="Variant name (full, market_hours, derived_market_hours, outside_market_hours)")
    parser.add_argument(
        "--sort-by",
        default="accuracy",
        choices=["accuracy", "pnl_pct", "daily_pnl_pct", "net_wins", "daily_net_wins"],
        help="Metric to sort the generated markdown table by.",
    )
    parser.add_argument(
        "--output-markdown-path",
        type=str,
        help="Optional path to write the generated markdown table to.",
    )
    parser.add_argument(
        "--group-by",
        default="auto",
        choices=["auto", "none", "family"],
        help="Optional grouping mode for the generated markdown output.",
    )
    parser.add_argument(
        "--update-experiments-md",
        action="store_true",
        default=True,
        help="Automatically update EXPERIMENTS.md with the generated tables",
    )
    parser.add_argument(
        "--no-update-experiments-md",
        action="store_false",
        dest="update_experiments_md",
        help="Do not update EXPERIMENTS.md; only print/write the generated table.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated table without updating EXPERIMENTS.md",
    )
    return parser.parse_args()


def load_summary_json(path: Path) -> dict[str, Any]:
    """Load and parse summary.json file."""
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detect_family(model_name: str) -> str:
    text = str(model_name or "").upper()
    for family in FAMILY_ORDER:
        if text.startswith(family):
            return family
    return text.split("_", 1)[0] if text else "UNKNOWN"


def extract_metrics_from_summary(
    summary: dict[str, Any],
    *,
    sort_by: str = "accuracy",
) -> list[dict[str, Any]]:
    """
    Extract model metrics from summary.json.

    Returns:
        List of dicts with keys: name, accuracy, pnl_pct, daily_pnl_pct, net_wins, daily_net_wins
    """
    variant_name = str(summary.get("variant", ""))
    test_rows = int(summary.get("dataset_metadata", {}).get("test_rows", 5000))
    days_in_test = test_rows / 24 if test_rows > 0 else 0.0
    shared_test_timestamps = summary.get("plotting", {}).get("test_timestamps", [])

    metrics = []
    for model_entry in summary.get("models", []):
        if not isinstance(model_entry, dict) or "name" not in model_entry:
            continue

        name = model_entry["name"]
        test_data = model_entry.get("test", {})
        portfolio_data = model_entry.get("portfolio", {})

        # Extract accuracy
        accuracy = test_data.get("accuracy")
        if accuracy is None:
            continue

        # Extract pnl_pct from portfolio.fixed_dollar section
        # pnl_pct is stored as percentage (e.g., 23.8 for 2380%)
        fixed_dollar_portfolio = portfolio_data.get("fixed_dollar", {})
        pnl_pct = fixed_dollar_portfolio.get("pnl_pct", 0.0)

        model_test_timestamps = test_data.get("timestamps", [])
        if not isinstance(model_test_timestamps, list) or not model_test_timestamps:
            model_test_timestamps = shared_test_timestamps if isinstance(shared_test_timestamps, list) else []

        active_days = days_in_test
        if variant_name in {"market_hours", "derived_market_hours", "outside_market_hours"} and model_test_timestamps:
            try:
                timestamps = pd.to_datetime(model_test_timestamps, utc=True)
                active_days = float(len({timestamp.tz_convert(EASTERN_TZ).date() for timestamp in timestamps}))
            except Exception:
                active_days = days_in_test
        if active_days <= 0:
            active_days = days_in_test if days_in_test > 0 else 1.0

        # Calculate daily pnl_pct by dividing total by number of days
        daily_pnl_pct = pnl_pct / active_days if active_days > 0 else 0.0

        # Count wins and losses from trade_pnls (1.0 = win, -1.0 = loss)
        trade_pnls = fixed_dollar_portfolio.get("trade_pnls", [])
        wins = sum(1 for pnl in trade_pnls if pnl > 0)
        losses = sum(1 for pnl in trade_pnls if pnl < 0)
        net_wins = wins - losses

        total_trades = len(trade_pnls)
        if total_trades <= 0:
            total_trades = int(test_data.get("accuracy_scored_count") or 0)
        if total_trades <= 0 and isinstance(model_test_timestamps, list):
            total_trades = len(model_test_timestamps)

        if variant_name in {"market_hours", "derived_market_hours", "outside_market_hours"}:
            daily_net_wins = net_wins / active_days if active_days > 0 else 0.0
        else:
            daily_net_wins = (net_wins * 24 / total_trades) if total_trades > 0 else 0.0

        metrics.append(
            {
                "name": name,
                "family": str(model_entry.get("family") or detect_family(name)),
                "accuracy": accuracy,
                "pnl_pct": pnl_pct,
                "daily_pnl_pct": daily_pnl_pct,
                "net_wins": net_wins,
                "daily_net_wins": daily_net_wins,
            }
        )

    sort_key_name = str(sort_by or "accuracy")
    metrics.sort(key=lambda metric: metric.get(sort_key_name, float("-inf")), reverse=True)
    return metrics


def format_percentage(value: float, multiply_by_100: bool = True) -> str:
    """
    Format a value as a percentage string with ± sign.
    
    Args:
        value: Value to format (e.g., 23.8 for pnl_pct stored as percentage)
        multiply_by_100: If True, multiply by 100 before formatting
    """
    display_value = value * 100 if multiply_by_100 else value
    if display_value >= 0:
        return f"+{display_value:.2f}%"
    else:
        return f"{display_value:.2f}%"


def format_accuracy(value: float) -> str:
    """Format accuracy as percentage."""
    return f"{value * 100:.2f}%"


def format_net_wins(value: int | float) -> str:
    """
    Format net wins as an integer with ± sign.
    
    Args:
        value: Net wins count (e.g., 10 or -5)
    """
    int_value = int(value)
    if int_value >= 0:
        return f"+{int_value}"
    else:
        return f"{int_value}"


def format_daily_net_wins(value: float) -> str:
    """
    Format daily net wins as a float with ± sign.
    
    Args:
        value: Daily net wins (e.g., 0.048 for +0.048)
    """
    if value >= 0:
        return f"+{value:.3f}"
    else:
        return f"{value:.3f}"


def generate_markdown_table(
    metrics: list[dict[str, Any]],
    variant_name: str,
) -> str:
    """Generate a markdown table from metrics."""
    lines = [f"**{variant_name}**\n"]
    lines.append("| Model | Accuracy | %PnL | Daily %PnL | Net Wins | Daily Net Wins |")
    lines.append("|---|---|---|---|---|---|")

    for metric in metrics:
        name = metric["name"]
        accuracy = format_accuracy(metric["accuracy"])
        # pnl_pct and daily_pnl_pct need to be multiplied by 100 for display
        pnl = format_percentage(metric["pnl_pct"], multiply_by_100=True)
        daily_pnl = format_percentage(metric["daily_pnl_pct"], multiply_by_100=True)
        net_wins = format_net_wins(metric["net_wins"])
        daily_net_wins = format_daily_net_wins(metric["daily_net_wins"])
        lines.append(f"| {name} | {accuracy} | {pnl} | {daily_pnl} | {net_wins} | {daily_net_wins} |")

    return "\n".join(lines)


def _family_sort_key(family: str) -> tuple[int, str]:
    normalized = str(family or "").upper()
    if normalized in FAMILY_ORDER:
        return (FAMILY_ORDER.index(normalized), normalized)
    return (len(FAMILY_ORDER), normalized)


def generate_grouped_markdown_table(
    metrics: list[dict[str, Any]],
    variant_name: str,
) -> str:
    lines = [f"**{variant_name}**"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        family = str(metric.get("family") or detect_family(metric.get("name", "")))
        grouped.setdefault(family, []).append(metric)

    for family in sorted(grouped.keys(), key=_family_sort_key):
        lines.append("")
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| Model | Accuracy | %PnL | Daily %PnL | Net Wins | Daily Net Wins |")
        lines.append("|---|---|---|---|---|---|")
        for metric in grouped[family]:
            name = metric["name"]
            accuracy = format_accuracy(metric["accuracy"])
            pnl = format_percentage(metric["pnl_pct"], multiply_by_100=True)
            daily_pnl = format_percentage(metric["daily_pnl_pct"], multiply_by_100=True)
            net_wins = format_net_wins(metric["net_wins"])
            daily_net_wins = format_daily_net_wins(metric["daily_net_wins"])
            lines.append(f"| {name} | {accuracy} | {pnl} | {daily_pnl} | {net_wins} | {daily_net_wins} |")

    return "\n".join(lines)


def should_group_by_family(
    *,
    group_by: str,
    experiment_num: int | None,
    summary_path: Path,
) -> bool:
    mode = str(group_by or "auto").lower()
    if mode == "family":
        return True
    if mode == "none":
        return False
    if experiment_num == 23:
        return True
    parts = [part.lower() for part in summary_path.parts]
    return "experiments" in parts and "23" in parts


def generate_all_variant_tables(
    experiment_dir: Path,
) -> dict[str, str]:
    """
    Generate tables for all available variants in an experiment.

    Returns:
        Dict mapping variant_name -> markdown_table_string
    """
    tables = {}
    for variant_subdir in experiment_dir.iterdir():
        if not variant_subdir.is_dir():
            continue
        summary_path = variant_subdir / "summary.json"
        if not summary_path.exists():
            continue

        variant_name = variant_subdir.name
        try:
            summary = load_summary_json(summary_path)
            metrics = extract_metrics_from_summary(summary)
            if metrics:
                display_name = variant_name.replace("_", " ").title()
                table = generate_markdown_table(metrics, display_name)
                tables[variant_name] = table
        except Exception as e:
            print(f"Warning: Failed to generate table for {variant_name}: {e}", file=sys.stderr)

    return tables


def get_experiment_description(
    experiments_file: Path,
    experiment_num: int,
) -> str | None:
    """Extract the description section of an experiment from EXPERIMENTS.md."""
    content = experiments_file.read_text(encoding="utf-8")
    
    # Find the experiment section
    section_pattern = rf"##\s+{experiment_num}\.\s+([^\n]+)"
    match = re.search(section_pattern, content)
    if not match:
        return None

    section_start = match.start()
    section_title = match.group(0)

    # Find the end of the description (stop at the first ** variant or next section)
    desc_end_pattern = rf"(\n##\s+|\n\*\*[A-Z])"
    remaining = content[section_start + len(section_title):]
    desc_match = re.search(desc_end_pattern, remaining)

    if desc_match:
        desc_end = section_start + len(section_title) + desc_match.start()
    else:
        desc_end = len(content)

    description = content[section_start:desc_end]
    return description


def update_experiments_md(
    experiment_num: int,
    variant_tables: dict[str, str],
    dry_run: bool = False,
) -> bool:
    """
    Update EXPERIMENTS.md with generated variant tables.

    Preserves the prose block (including ``- Results:`` bullets) before the first
    variant table, then replaces all auto-generated tables with ``variant_tables``.
    """
    experiments_file = PROJECT_ROOT / "EXPERIMENTS" / "EXPERIMENTS.md"
    if not experiments_file.exists():
        print(f"Error: EXPERIMENTS.md not found at {experiments_file}", file=sys.stderr)
        return False

    content = experiments_file.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    section_start_index: int | None = None
    section_end_index = len(lines)

    heading_pattern = re.compile(r"^##\s+(\d+)\.\s+")
    for index, line in enumerate(lines):
        match = heading_pattern.match(line)
        if not match:
            continue
        heading_number = int(match.group(1))
        if heading_number == experiment_num:
            section_start_index = index
            continue
        if section_start_index is not None and heading_number != experiment_num:
            section_end_index = index
            break

    if section_start_index is None:
        print(f"Warning: Experiment section {experiment_num} not found in EXPERIMENTS.md", file=sys.stderr)
        return False

    section_lines = lines[section_start_index:section_end_index]
    if not section_lines:
        print(f"Warning: Experiment section {experiment_num} is empty in EXPERIMENTS.md", file=sys.stderr)
        return False

    header = section_lines[0]
    body = "".join(section_lines[1:])

    marker = re.search(r"\n\*\*", body)
    if marker:
        body_before_tables = body[: marker.start()]
    else:
        body_before_tables = body

    description_part = (header + body_before_tables).rstrip() + "\n"

    preferred = ["full", "market_hours", "outside_market_hours", "derived_market_hours"]
    ordered_keys = sorted(
        variant_tables.keys(),
        key=lambda k: (preferred.index(k) if k in preferred else 99, k),
    )
    ordered_tables = [variant_tables[k] for k in ordered_keys]
    new_tables = "\n\n".join(ordered_tables).rstrip() + "\n"
    new_section_content = f"{description_part}\n{new_tables}\n"

    updated_lines = lines[:section_start_index] + [new_section_content] + lines[section_end_index:]
    updated_content = "".join(updated_lines)

    if dry_run:
        print("DRY RUN: Would update EXPERIMENTS.md with the following new section:\n")
        print(new_section_content)
        return True

    experiments_file.write_text(updated_content, encoding="utf-8")
    print(f"✓ Updated {experiments_file} with experiment {experiment_num} tables")
    return True


def main() -> None:
    args = parse_args()

    # Determine summary path
    if args.summary_path:
        summary_path = Path(args.summary_path)
        if not summary_path.is_absolute():
            summary_path = PROJECT_ROOT / summary_path
    elif args.experiment_num:
        summary_path = PROJECT_ROOT / "EXPERIMENTS" / str(args.experiment_num) / args.variant / "summary.json"
    else:
        print("Error: Either --summary-path or --experiment-num must be provided")
        sys.exit(1)

    # Load and generate tables
    try:
        summary = load_summary_json(summary_path)
        metrics = extract_metrics_from_summary(summary, sort_by=args.sort_by)

        if not metrics:
            print("Warning: No metrics found in summary.json", file=sys.stderr)
            return

        variant_display_name = args.variant.replace("_", " ").title()
        group_by_family = should_group_by_family(
            group_by=args.group_by,
            experiment_num=args.experiment_num,
            summary_path=summary_path,
        )
        table = (
            generate_grouped_markdown_table(metrics, variant_display_name)
            if group_by_family
            else generate_markdown_table(metrics, variant_display_name)
        )
        print(table)

        if args.output_markdown_path:
            output_markdown_path = Path(args.output_markdown_path)
            if not output_markdown_path.is_absolute():
                output_markdown_path = PROJECT_ROOT / output_markdown_path
            output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
            output_markdown_path.write_text(table + "\n", encoding="utf-8")
            print(f"Saved markdown table to {output_markdown_path}")

        # Update EXPERIMENTS.md if requested
        if args.update_experiments_md and args.experiment_num:
            update_experiments_md(
                args.experiment_num,
                {args.variant: table},
                dry_run=args.dry_run,
            )

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing summary: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

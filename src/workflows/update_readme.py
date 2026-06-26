from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.public.evaluation.sanitizer import assert_public_safe_text
from src.workflows.update_pages import (
    _all_experiment_rows,
    _best_row,
    _load_experiments,
    _load_production,
    _metric,
    _phase_status,
)


START_MARKER = "<!-- RESULTS_SNAPSHOT_START -->"
END_MARKER = "<!-- RESULTS_SNAPSHOT_END -->"


def _score(value: float) -> str:
    return f"{value:.4f}"


def _snapshot(root: Path) -> str:
    experiments = _load_experiments(root)
    production = _load_production(root)
    lines = [START_MARKER, "## Results Snapshot", ""]

    if experiments:
        lines.extend(["### Experiments", "", "| Market | Phase progress | Top validation BA |", "|---|---:|---:|"])
        for experiment in experiments:
            best = _best_row(_all_experiment_rows(experiment))
            lines.append(
                f"| {experiment['market_id']} | {_phase_status(experiment)} | "
                f"{_score(_metric(best.get('validation'), 'balanced_accuracy'))} |"
            )
    else:
        lines.extend(["### Experiments", "", "No public experiment results yet."])

    lines.extend(["", "### Production", ""])
    if production:
        lines.extend(["| Market | Winner | Weighted score |", "|---|---|---:|"])
        for artifact in production:
            metrics = artifact.get("delayed_metrics") or artifact.get("metrics") or {}
            lines.append(
                f"| {artifact['market_id']} | {artifact.get('winner') or '-'} | "
                f"{_score(_metric(metrics, 'weighted_score'))} |"
            )
    else:
        lines.append("No public production results yet.")

    lines.extend(["", END_MARKER])
    return "\n".join(lines)


def update_readme(root: Path | str = ".") -> Path:
    resolved = Path(root).resolve()
    readme = resolved / "README.md"
    text = readme.read_text(encoding="utf-8")
    snapshot = _snapshot(resolved)
    if START_MARKER in text and END_MARKER in text:
        before, remainder = text.split(START_MARKER, 1)
        _, after = remainder.split(END_MARKER, 1)
        text = before.rstrip() + "\n\n" + snapshot + after
    else:
        text = text.rstrip() + "\n\n" + snapshot + "\n"
    assert_public_safe_text(text)
    readme.write_text(text, encoding="utf-8")
    return readme


def main() -> int:
    update_readme(Path("."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

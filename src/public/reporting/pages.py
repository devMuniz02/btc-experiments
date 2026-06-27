from __future__ import annotations

from pathlib import Path

from src.public.evaluation.sanitizer import assert_public_safe_text


def write_pages_docs(root: Path, *, market_id: str, summary: dict[str, object]) -> list[Path]:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    index = docs / "index.md"
    experiments = docs / "experiments.md"
    prod = docs / "prod.md"
    index_text = f"# Quant ML Platform\n\nLatest market: {market_id}\n\nStatus: {summary.get('status', 'unknown')}\n"
    experiments_text = f"# Experiments\n\nAnonymized candidate count: {summary.get('candidate_count', 0)}\n"
    prod_text = f"# Production\n\nPublic delay hours: {summary.get('public_delay_hours', 24)}\n"
    for text in (index_text, experiments_text, prod_text):
        assert_public_safe_text(text)
    index.write_text(index_text, encoding="utf-8")
    experiments.write_text(experiments_text, encoding="utf-8")
    prod.write_text(prod_text, encoding="utf-8")
    return [index, experiments, prod]

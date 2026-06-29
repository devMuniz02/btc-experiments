from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.public.evaluation.sanitizer import assert_public_safe_text
from src.public.reporting.discovery_catalog import PUBLIC_DISCOVERY_PHASES


def _read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    return value if isinstance(value, type(default)) else default


def _market_from_path(path: Path) -> str:
    return path.parents[1].name


def _market_parts(market_id: str) -> tuple[str, str]:
    parts = market_id.rsplit("_", 1)
    return (parts[0].upper(), parts[1].upper()) if len(parts) == 2 else (market_id.upper(), "--")


def _market_label(market_id: str) -> str:
    ticker, interval = _market_parts(market_id)
    return f"{ticker} {interval}"


def _focus(value: str) -> str:
    text = str(value or "Pending").replace("_", " ").strip().lower()
    return text[:1].upper() + text[1:]


def _metric(metrics: Any, key: str) -> float:
    try:
        return float((metrics or {}).get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}" if math.isfinite(number) else "-"


def _metric_number(metrics: Any, key: str) -> str:
    return _number((metrics or {}).get(key)) if key in (metrics or {}) else "-"


def _date(value: Any) -> str:
    text = str(value or "-")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return text


def _load_experiments(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "experiments").glob("*/results/run_manifest_public.json")):
        manifest = _read_json(manifest_path, {})
        rows = _read_json(manifest_path.with_name("phase_results_public.json"), [])
        candidates = _read_json(manifest_path.with_name("candidates_public.json"), [])
        output.append(
            {
                "market_id": str(manifest.get("market_id") or _market_from_path(manifest_path)),
                "workflow_profile": str(manifest.get("workflow_profile") or "legacy_v1"),
                "manifest": manifest,
                "phase_rows": rows,
                "candidates": candidates,
            }
        )
    return output


def _load_production(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for artifact_path in sorted((root / "prod").glob("*/results/production_public.json")):
        artifact = _read_json(artifact_path, {})
        artifact["market_id"] = str(artifact.get("market_id") or _market_from_path(artifact_path))
        output.append(artifact)
    return output


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("selected_for_next_phase")]
    eligible = selected or [row for row in rows if str(row.get("status")) in {"worked", "completed"}] or rows
    return max(
        eligible,
        key=lambda row: (
            _metric(row.get("validation"), "direction_accuracy"),
            _metric(row.get("validation"), "balanced_accuracy"),
            _metric(row.get("validation"), "mcc"),
            _metric(row.get("validation"), "f1"),
            str(row.get("experiment_id") or ""),
        ),
        default={},
    )


def _all_experiment_rows(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(experiment["phase_rows"])
    if rows:
        return rows
    converted: list[dict[str, Any]] = []
    for candidate in experiment["candidates"]:
        converted.append(
            {
                "phase_id": "phase0",
                "axis": "candidate_evaluation",
                "status": candidate.get("status"),
                "train": candidate.get("train"),
                "validation": candidate.get("validation"),
                "test": candidate.get("test"),
                "overfit_diagnostics": {},
            }
        )
    return converted


def _phase_groups(experiment: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    rows = _all_experiment_rows(experiment)
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_phase.setdefault(str(row.get("phase_id") or "phase0"), []).append(row)
    return [
        (phase, group)
        for phase, group in sorted(by_phase.items(), key=lambda item: _phase_sort_key(item[0]))
    ]


def _phase_status(experiment: dict[str, Any]) -> str:
    groups = _phase_groups(experiment)
    if experiment["workflow_profile"] == "exhaustive_v1":
        complete = sum(bool(rows) for _, rows in groups)
        total = len(PUBLIC_DISCOVERY_PHASES)
        return "Complete" if complete == total else f"{complete}/{total}"
    return "Complete" if groups else "0/1"


def _latest_phase_best_row(experiment: dict[str, Any]) -> dict[str, Any]:
    for _, rows in reversed(_phase_groups(experiment)):
        if rows:
            return _best_row(rows)
    return {}


def _market_badges(market_id: str) -> str:
    ticker, interval = _market_parts(market_id)
    hue = sum(ord(char) for char in ticker) * 47 % 360
    digits = "".join(char for char in interval if char.isdigit())
    amount = int(digits or 1)
    unit = interval[-1:] if interval else "H"
    scale = amount * {"M": 1, "H": 60, "D": 1440, "W": 10080}.get(unit, 60)
    lightness = max(28, 62 - min(34, int(math.log2(max(1, scale)) * 4)))
    return (
        f'<span class="badge market" style="--badge-hue:{hue}">{escape(ticker)}</span>'
        f'<span class="badge interval" style="--interval-light:{lightness}%">{escape(interval)}</span>'
    )


def _table(headers: list[str], rows: list[list[str]], classes: str = "") -> str:
    header = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table class="{classes}"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def _status(value: str) -> str:
    style = "queued" if value.lower() == "queued" else "complete"
    return f'<span class="status {style}">{escape(value)}</span>'


def _link(label: str, href: str) -> str:
    return f'<a class="action" href="{escape(href)}">{escape(label)} <span aria-hidden="true">&#8594;</span></a>'


def _empty(title: str, text: str) -> str:
    return f'<div class="empty"><strong>{escape(title)}</strong><span>{escape(text)}</span></div>'


def _summary_cards(items: list[tuple[str, str]]) -> str:
    return '<section class="kpis">' + "".join(
        f'<article class="kpi"><span>{escape(label)}</span><strong>{escape(value)}</strong></article>'
        for label, value in items
    ) + "</section>"


def _hero(title: str, subtitle: str, active: str) -> str:
    links = "".join(
        f'<a class="{"active" if name == active else ""}" href="{href}">{label}</a>'
        for name, label, href in (
            ("overview", "Overview", "index.html"),
            ("experiments", "Experiments", "experiments.html"),
            ("production", "Production", "prod.html"),
        )
    )
    return f"""
<header class="topbar"><a class="brand" href="index.html">Quant Results</a><nav>{links}</nav><button id="theme-toggle" class="theme" type="button">Light mode</button></header>
<section class="hero"><p class="eyebrow">System performance</p><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></section>
"""


def _section(title: str, action: str = "", subtitle: str = "") -> str:
    detail = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    return f'<div class="section-head"><div><h2>{escape(title)}</h2>{detail}</div>{action}</div>'


def _overview(experiments: list[dict[str, Any]], production: list[dict[str, Any]]) -> str:
    experiment_rows = [_latest_phase_best_row(item) for item in experiments]
    completed = sum(sum(bool(rows) for _, rows in _phase_groups(item)) for item in experiments)
    total = sum(
        len(PUBLIC_DISCOVERY_PHASES) if item["workflow_profile"] == "exhaustive_v1" else len(_phase_groups(item))
        for item in experiments
    )
    top_valid = max((_metric(row.get("validation"), "balanced_accuracy") for row in experiment_rows), default=0.0)
    top_prod = max((_metric(item.get("delayed_metrics"), "weighted_score") for item in production), default=0.0)
    cards = _summary_cards(
        [
            ("Experiment markets", str(len(experiments))),
            ("Completed phases", f"{completed}/{total}" if total else "0/0"),
            ("Top validation BA", _number(top_valid)),
            ("Top production score", _number(top_prod)),
        ]
    )
    winners = _table(
        ["Market", "Interval", "Winner", "Generated UTC", "Latest Public Window", "Weighted Score", "Direction Accuracy"],
        [
            [
                _market_badges(item["market_id"]).split('</span>')[0] + '</span>',
                _market_badges(item["market_id"]).split('</span>')[1] + '</span>',
                f'<span class="model-id">{escape(_production_winner_label(item))}</span>',
                escape(_date(item.get("generated_at_utc"))),
                escape(_date(item.get("latest_public_window"))),
                escape(_number(_metric(item.get("delayed_metrics"), "weighted_score"))),
                escape(_number(_metric(item.get("delayed_metrics"), "direction_accuracy"))),
            ]
            for item in production
        ],
    ) if production else _empty("No production results yet", "Promoted model summaries will appear here.")
    experiment_table = _table(
        ["Market", "Interval", "Phase Status", "Latest Phase BA", "Latest Direction Accuracy", "Latest MCC", "Latest Weighted Score"],
        [
            [
                _market_badges(item["market_id"]).split('</span>')[0] + '</span>',
                _market_badges(item["market_id"]).split('</span>')[1] + '</span>',
                _status(_phase_status(item)),
                escape(_number(_metric(best.get("validation"), "balanced_accuracy"))),
                escape(_number(_metric(best.get("validation"), "direction_accuracy"))),
                escape(_number(_metric(best.get("validation"), "mcc"))),
                escape(_number(_metric(best.get("validation"), "weighted_score"))),
            ]
            for item, best in zip(experiments, experiment_rows, strict=True)
        ],
    ) if experiments else _empty("No experiment results yet", "Candidate summaries will appear here.")
    return "".join(
        [
            _hero("Market intelligence dashboard", "Validation-ranked research and promoted model performance.", "overview"),
            cards,
            _section("Production Winners", _link("Open production", "prod.html")), winners,
            _section("Experiment Candidates", _link("Open experiments", "experiments.html")), experiment_table,
        ]
    )


def _phase_number(phase_id: str) -> str:
    digits = "".join(char for char in phase_id if char.isdigit())
    return str(int(digits)) if digits else phase_id


def _phase_sort_key(phase_id: str) -> tuple[int, str]:
    digits = "".join(char for char in str(phase_id) if char.isdigit())
    return (int(digits) if digits else 9999, str(phase_id))


def _experiment_table(experiment: dict[str, Any], market_index: int) -> str:
    rows: list[str] = []
    for phase_id, candidates in _phase_groups(experiment):
        best = _best_row(candidates)
        group = f"m{market_index}-{phase_id}"
        has_candidates = bool(candidates)
        status = str(best.get("status") or ("queued" if not candidates else "completed"))
        selected = "Yes" if best.get("selected_for_next_phase") else ("-" if not candidates else "No")
        toggle = (
            f'<button class="phase-toggle" type="button" data-group="{escape(group)}" aria-expanded="false">{escape(_phase_number(phase_id))}</button>'
            if has_candidates else escape(_phase_number(phase_id))
        )
        metrics = [best.get("train"), best.get("validation"), best.get("test")]
        gap = (best.get("overfit_diagnostics") or {}).get("train_valid_gap")
        parent = [
            toggle, _status(_focus(status)), escape(selected),
            escape(_number(_metric(metrics[0], "balanced_accuracy"))) if best else "-",
            escape(_number(_metric(metrics[0], "direction_accuracy"))) if best else "-",
            escape(_number(_metric(metrics[1], "balanced_accuracy"))) if best else "-",
            escape(_number(_metric(metrics[1], "direction_accuracy"))) if best else "-",
            escape(_number(_metric(metrics[2], "balanced_accuracy"))) if best else "-",
            escape(_number(_metric(metrics[2], "direction_accuracy"))) if best else "-",
            escape(_number(_metric(metrics[1], "mcc"))) if best else "-",
            escape(_number(_metric(metrics[1], "weighted_score"))) if best else "-",
            escape(_number(best.get("runtime_seconds"), 2)) if best and best.get("runtime_seconds") is not None else "-",
            escape(_number(gap)) if gap is not None else "-",
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in parent) + "</tr>")
        ordered = sorted(
            candidates,
            key=lambda row: (
                -_metric(row.get("validation"), "direction_accuracy"),
                -_metric(row.get("validation"), "balanced_accuracy"),
                -_metric(row.get("validation"), "mcc"),
                str(row.get("experiment_id") or ""),
            ),
        )
        for index, candidate in enumerate(ordered):
            label = chr(65 + index) if index < 26 else f"A{index - 25}"
            gap = (candidate.get("overfit_diagnostics") or {}).get("train_valid_gap")
            expanded = [
                escape(label), _status(_focus(str(candidate.get("status") or "completed"))),
                "Yes" if candidate.get("selected_for_next_phase") else "-",
                escape(_number(_metric(candidate.get("train"), "balanced_accuracy"))),
                escape(_number(_metric(candidate.get("train"), "direction_accuracy"))),
                escape(_number(_metric(candidate.get("validation"), "balanced_accuracy"))),
                escape(_number(_metric(candidate.get("validation"), "direction_accuracy"))),
                escape(_number(_metric(candidate.get("test"), "balanced_accuracy"))),
                escape(_number(_metric(candidate.get("test"), "direction_accuracy"))),
                escape(_number(_metric(candidate.get("validation"), "mcc"))),
                escape(_number(_metric(candidate.get("validation"), "weighted_score"))),
                escape(_number(candidate.get("runtime_seconds"), 2)) if candidate.get("runtime_seconds") is not None else "-",
                escape(_number(gap)) if gap is not None else "-",
            ]
            rows.append(f'<tr class="candidate-row" data-group="{escape(group)}" hidden>' + "".join(f"<td>{cell}</td>" for cell in expanded) + "</tr>")
    headers = ["Phase", "Status", "Selected", "Train BA", "Train Direction", "Valid BA", "Valid Direction", "Test BA", "Test Direction", "Valid MCC", "Valid Weighted", "Runtime", "Train-Valid Gap"]
    return f'<div class="table-wrap"><table class="phase-table"><thead><tr>{"".join(f"<th>{escape(h)}</th>" for h in headers)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _post_lock_sliding_table(experiment: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    phase16_rows = [row for row in _all_experiment_rows(experiment) if str(row.get("phase_id")) == "phase16"]
    rows_with_summary = [
        row
        for row in phase16_rows
        if (row.get("post_lock_sliding_summary") or {}).get("window_metrics_summary")
        and int((row.get("post_lock_sliding_summary") or {}).get("windows_completed") or 0) > 0
    ]
    selected_rows = [row for row in rows_with_summary if row.get("selected_for_next_phase")]
    display_rows = selected_rows or rows_with_summary
    display_rows = sorted(
        display_rows,
        key=lambda item: (
            -_metric(item.get("validation"), "direction_accuracy"),
            -_metric(item.get("validation"), "balanced_accuracy"),
            str(item.get("experiment_id") or ""),
        ),
    )[:3]
    for index, row in enumerate(
        display_rows
    ):
        summary = row.get("post_lock_sliding_summary") or {}
        label = str(summary.get("model_public_id") or (chr(65 + index) if index < 26 else f"A{index - 25}"))
        metrics = summary.get("window_metrics_summary") or {}
        direction = metrics.get("direction_accuracy") or {}
        balanced = metrics.get("balanced_accuracy") or {}
        long_run = summary.get("long_run_metrics") or {}
        rows.append(
            [
                f'<span class="model-id">{escape(label)}</span>',
                escape(str(summary.get("windows_completed") or "-")),
                escape(_number(direction.get("min"))),
                escape(_number(direction.get("avg"))),
                escape(_number(direction.get("max"))),
                escape(_number(balanced.get("min"))),
                escape(_number(balanced.get("avg"))),
                escape(_number(balanced.get("max"))),
                escape(_number(_metric(long_run, "direction_accuracy"))),
                escape(_number(_metric(long_run, "balanced_accuracy"))),
            ]
        )
    if not rows:
        return ""
    return _section(f"{_market_label(experiment['market_id'])} Post-Lock Sliding Validation", subtitle="Out-of-sample 300-row test windows before the production anchor, plus first-window long-run test metrics.") + _table(
        [
            "Model",
            "Windows",
            "Min Direction",
            "Avg Direction",
            "Max Direction",
            "Min BA",
            "Avg BA",
            "Max BA",
            "Long Direction",
            "Long BA",
        ],
        rows,
    )


def _experiments_page(experiments: list[dict[str, Any]]) -> str:
    if not experiments:
        content = _empty("No experiment results yet", "Phase progress will appear after the first run.")
    else:
        blocks = []
        for index, experiment in enumerate(experiments):
            label = _market_label(experiment["market_id"])
            blocks.append(
                f'<section class="market-block">{_section(f"{label} Experiment Progress", _status(_phase_status(experiment)))}'
                f'<div class="market-title">{_market_badges(experiment["market_id"])}</div>'
                f'{_experiment_table(experiment, index)}{_post_lock_sliding_table(experiment)}</section>'
            )
        content = "".join(blocks)
    return _hero("Experiments", "Every discovery phase, with candidate detail available in place.", "experiments") + content


def _series_points(series: Any) -> list[tuple[datetime, float]]:
    points: list[tuple[datetime, float]] = []
    for item in series or []:
        try:
            points.append((datetime.fromisoformat(str(item["timestamp"]).replace("Z", "+00:00")), float(item["performance"])))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(points)


def _plot(series: list[tuple[str, Any]]) -> str:
    parsed = [(label, _series_points(points)) for label, points in series]
    parsed = [(label, points) for label, points in parsed if points]
    if not parsed:
        return ""
    width, height = 920, 330
    left, right, top, bottom = 72, 24, 28, 58
    all_points = [point for _, points in parsed for point in points]
    min_time = min(point[0].timestamp() for point in all_points)
    max_time = max(point[0].timestamp() for point in all_points)
    values = [point[1] for point in all_points]
    min_value, max_value = min(values), max(values)
    if min_value == max_value:
        min_value -= 0.01
        max_value += 0.01
    time_span = max(max_time - min_time, 1.0)
    value_span = max_value - min_value
    chart_width, chart_height = width - left - right, height - top - bottom
    colors = ("#38d6b4", "#f6b84a", "#ff6f7d", "#8ca7ff", "#c98cff", "#7fd15d")
    grid: list[str] = []
    for index in range(5):
        y = top + chart_height * index / 4
        value = max_value - value_span * index / 4
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/><text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.3f}</text>')
    for index in range(4):
        x = left + chart_width * index / 3
        stamp = datetime.fromtimestamp(min_time + time_span * index / 3, tz=all_points[0][0].tzinfo)
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/><text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle">{stamp:%Y-%m-%d}</text>')
    lines: list[str] = []
    legend: list[str] = []
    for index, (label, points) in enumerate(parsed):
        color = colors[index % len(colors)]
        coordinates = " ".join(
            f"{left + (stamp.timestamp()-min_time)/time_span*chart_width:.1f},{top + (max_value-value)/value_span*chart_height:.1f}"
            for stamp, value in points
        )
        lines.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5" vector-effect="non-scaling-stroke"/>')
        legend.append(f'<span><i style="--series:{color}"></i>{escape(label)}</span>')
    svg = (
        f'<div class="chart"><div class="legend">{"".join(legend)}</div><svg viewBox="0 0 {width} {height}" role="img" aria-label="Performance over time">'
        f'<g class="axis">{"".join(grid)}{"".join(lines)}<text x="{width/2}" y="{height-8}" text-anchor="middle" class="axis-label">UTC date</text>'
        f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" class="axis-label">Performance</text></g></svg></div>'
    )
    return svg


def _top_models(production: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for market in production:
        models = market.get("top_models") or []
        if not models and market.get("winner"):
            models = [{"rank": 1, "public_id": market["winner"], "delayed_metrics": market.get("delayed_metrics") or {}, "prediction_series": market.get("prediction_series") or []}]
        rows.extend((market, model) for model in models)
    return sorted(rows, key=lambda item: (-_metric(item[1].get("delayed_metrics"), "weighted_score"), item[0]["market_id"], int(item[1].get("rank") or 99)))


def _production_winner_label(item: dict[str, Any]) -> str:
    for model in item.get("top_models") or []:
        if model.get("public_id"):
            return str(model["public_id"])
    for model_id in item.get("production_model_set") or []:
        if model_id:
            return str(model_id)
    return str(item.get("winner") or "-")


def _bar(label: str, value: float) -> str:
    width = max(0.0, min(100.0, value * 100.0 if label != "MCC" else (value + 1.0) * 50.0))
    return f'<div class="metric-bar"><div><span>{escape(label)}</span><strong>{escape(_number(value))}</strong></div><i><b style="width:{width:.2f}%"></b></i></div>'


def _production_model_table(models: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    headers = [
        "Market", "Interval", "Model", "Version", "Active From", "Active Until",
        "Train BA", "Valid BA", "Production BA", "Weighted Score", "Direction Accuracy", "MCC",
    ]
    rows: list[str] = []
    for index, (market, model) in enumerate(models):
        group = f"prod-model-{index}"
        versions = list(model.get("versions") or [])
        current_version = int(model.get("current_version") or (versions[-1].get("version", 1) if versions else 1))
        metrics = model.get("delayed_metrics") or {}
        cells = [
            _market_badges(market["market_id"]).split('</span>')[0] + '</span>',
            _market_badges(market["market_id"]).split('</span>')[1] + '</span>',
            (
                f'<button class="phase-toggle" type="button" data-group="{escape(group)}" aria-expanded="false">'
                f'{escape(str(model.get("public_id") or "-"))}</button>'
                if versions else f'<span class="model-id">{escape(str(model.get("public_id") or "-"))}</span>'
            ),
            f"v{current_version}", "-", "-",
            escape(_metric_number(model.get("train"), "balanced_accuracy")),
            escape(_metric_number(model.get("validation"), "balanced_accuracy")),
            escape(_metric_number(metrics, "balanced_accuracy")),
            escape(_metric_number(metrics, "weighted_score")),
            escape(_metric_number(metrics, "direction_accuracy")),
            escape(_metric_number(metrics, "mcc")),
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        for version in versions:
            version_metrics = version.get("production_metrics") or {}
            expanded = [
                "-", "-", "-", f"v{int(version.get('version', 1))}",
                escape(_date(version.get("activated_at_utc"))),
                escape(_date(version.get("deactivated_at_utc"))) if version.get("deactivated_at_utc") else "Active",
                escape(_metric_number(version.get("train"), "balanced_accuracy")),
                escape(_metric_number(version.get("validation"), "balanced_accuracy")),
                escape(_metric_number(version_metrics, "balanced_accuracy")),
                escape(_metric_number(version_metrics, "weighted_score")),
                escape(_metric_number(version_metrics, "direction_accuracy")),
                escape(_metric_number(version_metrics, "mcc")),
            ]
            rows.append(
                f'<tr class="candidate-row" data-group="{escape(group)}" hidden>'
                + "".join(f"<td>{cell}</td>" for cell in expanded) + "</tr>"
            )
    return (
        f'<div class="table-wrap"><table><thead><tr>{"".join(f"<th>{escape(header)}</th>" for header in headers)}'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def _market_version_series(item: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    series: list[tuple[str, list[dict[str, Any]]]] = []
    for model in item.get("top_models") or []:
        versions = list(model.get("versions") or [])
        if versions:
            series.extend(
                (
                    f"{model.get('public_id') or 'Model'} v{int(version.get('version', 1))}",
                    list(version.get("prediction_series") or []),
                )
                for version in versions
                if version.get("prediction_series")
            )
        elif model.get("prediction_series"):
            series.append((str(model.get("public_id") or "Model"), list(model["prediction_series"])))
    return series


def _production_page(production: list[dict[str, Any]]) -> str:
    if not production:
        return _hero("Production", "Promoted model performance across public reporting windows.", "production") + _empty("No production results yet", "Production summaries will appear after promotion.")
    models = _top_models(production)
    best_market, best_model = models[0]
    best_metrics = best_model.get("delayed_metrics") or {}
    cards = _summary_cards(
        [
            ("Production markets", str(len(production))),
            ("Top K models", "K = 3"),
            ("UTC date", _date(max(str(item.get("generated_at_utc") or "") for item in production))),
            ("Top accuracy", _number(max(_metric(model.get("delayed_metrics"), "direction_accuracy") for _, model in models))),
        ]
    )
    best = (
        '<section class="best-model"><div><span class="label">Best model</span><div class="best-id">'
        f'{_market_badges(best_market["market_id"])}<strong>{escape(str(best_model.get("public_id") or "-"))} v{int(best_model.get("current_version") or 1)}</strong></div></div>'
        f'<div class="bars">{_bar("Weighted score", _metric(best_metrics, "weighted_score"))}{_bar("Direction accuracy", _metric(best_metrics, "direction_accuracy"))}{_bar("MCC", _metric(best_metrics, "mcc"))}</div></section>'
    )
    winners = _table(
        ["Market", "Interval", "Winner", "Generated UTC", "Latest Public Window", "Weighted Score", "Direction Accuracy"],
        [[
            _market_badges(item["market_id"]).split('</span>')[0] + '</span>',
            _market_badges(item["market_id"]).split('</span>')[1] + '</span>',
            f'<span class="model-id">{escape(_production_winner_label(item))}</span>',
            escape(_date(item.get("generated_at_utc"))), escape(_date(item.get("latest_public_window"))),
            escape(_number(_metric(item.get("delayed_metrics"), "weighted_score"))),
            escape(_number(_metric(item.get("delayed_metrics"), "direction_accuracy"))),
        ] for item in production],
    )
    model_table = _production_model_table(models)
    merged = _plot([
        (f'{_market_parts(item["market_id"])[0]} {_market_parts(item["market_id"])[1]} {_production_winner_label(item)}', item.get("prediction_series") or [])
        for item in production
    ])
    market_plots = "".join(
        f'<section class="market-block">{_section(_market_parts(item["market_id"])[0] + " " + _market_parts(item["market_id"])[1])}'
        f'{_plot(_market_version_series(item))}</section>'
        for item in production if _market_version_series(item)
    )
    plot_section = (_section("Market performance") + merged) if merged else ""
    return "".join([
        _hero("Production", "Promoted model performance across public reporting windows.", "production"), cards, best,
        _section("Production Winners"), winners, _section("Top K Models", subtitle="Ordered by production weighted score."), model_table,
        plot_section, market_plots,
    ])


def _markdown(experiments: list[dict[str, Any]], production: list[dict[str, Any]], page: str) -> str:
    if page == "index":
        return f"# Quant Results Dashboard\n\nExperiment markets: {len(experiments)}\n\nProduction markets: {len(production)}\n\n[Experiments](experiments.html) | [Production](prod.html)\n"
    if page == "experiments":
        lines = ["# Experiments", ""]
        if not experiments:
            return "# Experiments\n\nNo public runs yet.\n"
        lines.extend(f"- {_market_parts(item['market_id'])[0]} {_market_parts(item['market_id'])[1]}: {_phase_status(item)}" for item in experiments)
        return "\n".join(lines) + "\n"
    if not production:
        return "# Production\n\nNo public runs yet.\n"
    return "# Production\n\n" + "\n".join(
        f"- {_market_parts(item['market_id'])[0]} {_market_parts(item['market_id'])[1]}: {_production_winner_label(item)}"
        for item in production
    ) + "\n"


def _css() -> str:
    return """
:root{color-scheme:dark;--bg:#091116;--panel:#111d24;--panel2:#16252e;--ink:#eef7f5;--muted:#91a5aa;--line:#29404a;--accent:#38d6b4;--amber:#f6b84a;--red:#ff6f7d;--blue:#8ca7ff;--shadow:0 14px 38px rgba(0,0,0,.22)}
:root[data-theme="light"]{color-scheme:light;--bg:#eef3f2;--panel:#fff;--panel2:#f4f8f7;--ink:#142225;--muted:#617277;--line:#cddbd9;--accent:#087f6a;--amber:#a86700;--red:#c53f50;--blue:#4969bf;--shadow:0 12px 30px rgba(35,65,66,.1)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.45}a{color:inherit;text-decoration:none}.shell{max-width:1420px;margin:auto;padding:0 24px 56px}.topbar{height:68px;display:flex;align-items:center;border-bottom:1px solid var(--line);gap:28px}.brand{font-weight:850;font-size:18px}.topbar nav{display:flex;gap:6px;margin-right:auto}.topbar nav a,.theme{border:1px solid transparent;border-radius:6px;padding:8px 11px;color:var(--muted);font:inherit;font-weight:700;background:transparent}.topbar nav a:hover,.topbar nav a.active{color:var(--ink);background:var(--panel)}.theme{border-color:var(--line);color:var(--ink);cursor:pointer}.hero{padding:52px 0 30px;max-width:900px}.eyebrow{color:var(--accent)!important;text-transform:uppercase;font-size:12px;font-weight:900}.hero h1{font-size:clamp(38px,6vw,68px);line-height:1;margin:9px 0 16px;letter-spacing:0}.hero p{color:var(--muted);font-size:18px;margin:0}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:34px}.kpi{padding:18px;border:1px solid var(--line);border-radius:7px;background:var(--panel);box-shadow:var(--shadow);min-height:104px}.kpi:nth-child(2){border-top:3px solid var(--amber)}.kpi:nth-child(3){border-top:3px solid var(--accent)}.kpi:nth-child(4){border-top:3px solid var(--blue)}.kpi span,.label{display:block;color:var(--muted);font-size:12px;font-weight:800;text-transform:uppercase}.kpi strong{display:block;font-size:26px;margin-top:10px;overflow-wrap:anywhere}.section-head{display:flex;align-items:end;justify-content:space-between;gap:16px;margin:34px 0 12px}.section-head h2{font-size:22px;margin:0}.section-head p{margin:4px 0 0;color:var(--muted)}.action{color:var(--accent);font-weight:800}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:7px;background:var(--panel);box-shadow:var(--shadow)}table{border-collapse:collapse;width:100%;min-width:900px}th,td{padding:13px 14px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-size:11px;text-transform:uppercase;background:var(--panel2)}tr:last-child td{border-bottom:0}tbody tr:hover{background:color-mix(in srgb,var(--panel2) 72%,transparent)}.badge{display:inline-flex;align-items:center;justify-content:center;border:1px solid var(--line);border-radius:5px;padding:4px 8px;font-size:11px;font-weight:900}.badge.market{color:hsl(var(--badge-hue) 76% 69%);background:hsl(var(--badge-hue) 45% 20%)}:root[data-theme="light"] .badge.market{color:hsl(var(--badge-hue) 65% 30%);background:hsl(var(--badge-hue) 65% 92%)}.badge.interval{color:#fff;background:hsl(205 65% var(--interval-light))}.market-title{display:flex;gap:6px;margin:0 0 12px}.status{display:inline-block;border-radius:999px;padding:4px 9px;background:rgba(56,214,180,.12);color:var(--accent);font-size:11px;font-weight:850}.status.queued{background:rgba(145,165,170,.12);color:var(--muted)}.model-id{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--accent);font-weight:800}.empty{padding:30px;border:1px dashed var(--line);border-radius:7px;background:var(--panel)}.empty strong,.empty span{display:block}.empty strong{font-size:20px}.empty span{color:var(--muted);margin-top:5px}.market-block{margin:30px 0}.phase-toggle{font:inherit;font-weight:900;color:var(--accent);border:0;background:transparent;cursor:pointer;padding:0}.phase-toggle:after{content:" +"}.phase-toggle[aria-expanded="true"]:after{content:" -"}.candidate-row{background:color-mix(in srgb,var(--accent) 4%,var(--panel))}.best-model{display:grid;grid-template-columns:minmax(240px,.7fr) 1.3fr;gap:32px;padding:22px;background:var(--panel);border:1px solid var(--line);border-radius:7px;box-shadow:var(--shadow)}.best-id{display:flex;align-items:center;gap:7px;margin-top:12px}.best-id strong{font-size:28px;margin-left:8px}.bars{display:grid;gap:12px}.metric-bar>div{display:flex;justify-content:space-between}.metric-bar span{color:var(--muted)}.metric-bar i{display:block;height:7px;margin-top:7px;background:var(--panel2);border-radius:3px;overflow:hidden}.metric-bar b{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--amber))}.chart{padding:14px;background:var(--panel);border:1px solid var(--line);border-radius:7px;box-shadow:var(--shadow)}.chart svg{display:block;width:100%;height:auto}.axis text{fill:var(--muted);font-size:11px}.axis .axis-label{fill:var(--ink);font-size:12px;font-weight:800}.grid{stroke:var(--line);stroke-width:1}.legend{display:flex;flex-wrap:wrap;gap:14px;padding:3px 8px 8px;color:var(--muted);font-size:12px}.legend span{display:flex;align-items:center;gap:6px}.legend i{width:18px;height:3px;background:var(--series)}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}.best-model{grid-template-columns:1fr}.topbar{gap:12px}.topbar nav{overflow:auto}.shell{padding:0 14px 42px}}@media(max-width:560px){.kpis{grid-template-columns:1fr}.brand{display:none}.hero{padding-top:34px}.topbar nav a{padding:7px}.theme{font-size:12px}}
""".strip()


def _document(title: str, body: str) -> str:
    script = """
<script>
const root=document.documentElement;const button=document.getElementById('theme-toggle');
const saved=localStorage.getItem('quant-theme');if(saved==='light'){root.dataset.theme='light';button.textContent='Dark mode';}
button.addEventListener('click',()=>{const light=root.dataset.theme!=='light';root.dataset.theme=light?'light':'dark';button.textContent=light?'Dark mode':'Light mode';localStorage.setItem('quant-theme',light?'light':'dark');});
document.querySelectorAll('.phase-toggle').forEach(button=>button.addEventListener('click',()=>{const open=button.getAttribute('aria-expanded')==='true';button.setAttribute('aria-expanded',String(!open));document.querySelectorAll('.candidate-row[data-group="'+button.dataset.group+'"]').forEach(row=>row.hidden=open);}));
</script>
"""
    return f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{_css()}</style></head><body><main class="shell">{body}</main>{script}</body></html>\n'


def update_pages(root: Path | str = ".") -> list[Path]:
    resolved = Path(root).resolve()
    docs = resolved / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    experiments = _load_experiments(resolved)
    production = _load_production(resolved)
    outputs = {
        docs / "index.md": _markdown(experiments, production, "index"),
        docs / "experiments.md": _markdown(experiments, production, "experiments"),
        docs / "prod.md": _markdown(experiments, production, "prod"),
        docs / "index.html": _document("Quant Results", _overview(experiments, production)),
        docs / "experiments.html": _document("Experiments", _experiments_page(experiments)),
        docs / "prod.html": _document("Production", _production_page(production)),
    }
    for path, text in outputs.items():
        assert_public_safe_text(text)
        path.write_text(text, encoding="utf-8")
    return list(outputs)


def main() -> int:
    update_pages(Path("."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

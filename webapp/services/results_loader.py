from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def load_method_comparison() -> list[dict[str, Any]]:
    """Combine available summary files + Fair CVAE log summary for Results page."""
    summary_path = ROOT / "tech_adversarial_grl_summary.csv"
    summary_df = _read_csv_if_exists(summary_path)

    rows: list[dict[str, Any]] = []
    if summary_df is not None:
        for _, row in summary_df.iterrows():
            method = str(row["Method"])
            label_map = {
                "Baseline": "Baseline MLP",
                "Gender_Adversarial": "Standalone adversarial baseline (c.py, gender)",
                "Intersectional_Adversarial": "Standalone adversarial baseline (c.py, intersectional)",
            }
            rows.append(
                {
                    "method": label_map.get(method, method),
                    "accuracy": float(row.get("Accuracy", float("nan"))),
                    "f1": float(row.get("F1_Score", float("nan"))),
                    "min_di": float(row.get("Min_DI", float("nan"))),
                    "source": "c.py summary",
                }
            )

    # Feature-removal baseline + Fair CVAE variants from experiment log.
    rows.extend(_parse_cvae_and_ablation_from_log(ROOT / "try.txt"))
    return rows


def _parse_cvae_and_ablation_from_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    out: list[dict[str, Any]] = []

    # Feature-removal ablation table line.
    ablation_match = re.search(
        r"Remove biased \(8 features\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text
    )
    if ablation_match:
        out.append(
            {
                "method": "Feature-removal baseline",
                "accuracy": float(ablation_match.group(1)),
                "f1": float(ablation_match.group(2)),
                "min_di": float(ablation_match.group(3)),
                "source": "try.txt ablation block",
            }
        )

    for mode, label in [
        ("adv_only", "Fair CVAE adv_only"),
        ("no_adv", "Fair CVAE no_adv"),
        ("full", "Fair CVAE full"),
    ]:
        pattern = rf"\[{mode}\].*?Acc=([0-9.]+)\s*\|\s*F1=([0-9.]+)"
        match = re.search(pattern, text)
        if match:
            section = _extract_block_after(text, match.start())
            min_di = _extract_min_di(section)
            out.append(
                {
                    "method": label,
                    "accuracy": float(match.group(1)),
                    "f1": float(match.group(2)),
                    "min_di": min_di,
                    "source": "try.txt run log",
                }
            )

    return out


def _extract_block_after(text: str, start: int, max_chars: int = 1500) -> str:
    return text[start : start + max_chars]


def _extract_min_di(block: str) -> float | None:
    dis = [float(v) for v in re.findall(r"\s([0-9]\.[0-9]{3,4})\s+[\-0-9]\.[0-9]{3,4}", block)]
    return min(dis) if dis else None


def load_per_group_di_comparison() -> list[dict[str, Any]]:
    di_path = ROOT / "tech_adversarial_grl_di_comparison.csv"
    di_df = _read_csv_if_exists(di_path)
    if di_df is None:
        return []
    return di_df.to_dict(orient="records")


def load_latent_summary() -> dict[str, Any]:
    text_path = ROOT / "try.txt"
    if not text_path.exists():
        return {"rows": [], "note": "No latent-space log found."}

    text = text_path.read_text(encoding="utf-8", errors="ignore")
    section_match = re.search(
        r"Metric\s+Baseline\s+Fair CVAE\s+Reduction\s+\n\s*-+\n(?P<body>.*?)(\n\n|\n\[3\])",
        text,
        flags=re.S,
    )
    rows = []
    if section_match:
        for line in section_match.group("body").splitlines():
            line = line.strip()
            m = re.match(r"(.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+%)", line)
            if m:
                rows.append(
                    {
                        "metric": m.group(1).strip(),
                        "baseline": float(m.group(2)),
                        "fair_cvae": float(m.group(3)),
                        "reduction": m.group(4),
                    }
                )

    return {
        "rows": rows,
        "note": "Latent-space summary parsed from existing training log (try.txt).",
    }

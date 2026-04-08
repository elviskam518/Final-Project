from __future__ import annotations

import uuid
from pathlib import Path
<<<<<<< HEAD
from typing import Any, Callable
=======
from typing import Any
>>>>>>> origin/main

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from b import add_proxy_qualified, compute_fairness_metrics, compute_odds_ratios
from c import (
    AdversarialDebiasingGRL,
    SimpleClassifier,
    evaluate_model,
    load_and_prepare_data,
    train_adversarial_model_grl,
    train_baseline_model,
)
<<<<<<< HEAD
from .cvae_runner import run_fair_cvae_mode
=======
from .results_loader import load_method_comparison
>>>>>>> origin/main


ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = ROOT / "webapp" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_upload(file_bytes: bytes, filename: str) -> str:
    upload_id = str(uuid.uuid4())
    suffix = Path(filename).suffix or ".csv"
    target = UPLOAD_DIR / f"{upload_id}{suffix}"
    target.write_bytes(file_bytes)
    return upload_id


def get_upload_path(upload_id: str) -> Path:
    matches = list(UPLOAD_DIR.glob(f"{upload_id}.*"))
    if not matches:
        raise FileNotFoundError(f"Upload id not found: {upload_id}")
    return matches[0]


def run_intermediate_analysis(csv_path: Path, top_quantile: float = 0.40) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    required = {"Gender", "Race", "Hired"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Group"] = df["Gender"].astype(str) + "_" + df["Race"].astype(str)
    df2 = add_proxy_qualified(df, top_quantile=top_quantile)

    baseline_group = "Male_White" if "Male_White" in set(df2["Group"]) else sorted(df2["Group"].unique())[0]
    fairness_df = compute_fairness_metrics(df2, baseline_group=baseline_group)
    odds_df = compute_odds_ratios(df2, baseline_group=baseline_group)
    odds_df = odds_df[odds_df["Term"].astype(str).str.startswith("G_")].copy()

    return {
        "baseline_group": baseline_group,
        "row_count": int(len(df2)),
        "groups": sorted(df2["Group"].unique().tolist()),
        "fairness": fairness_df.round(4).to_dict(orient="records"),
        "odds": odds_df.round(4).to_dict(orient="records"),
    }


<<<<<<< HEAD
def run_selected_model(
    csv_path: Path,
    model_key: str,
    run_latent: bool,
    log: Callable[[str], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    log_fn = log or (lambda _: None)
    progress_fn = progress or (lambda *_: None)

    if model_key in {"baseline_mlp", "standalone_adv"}:
        return _run_live_torch_model(csv_path, model_key, log_fn, progress_fn)

    mode_map = {
        "fair_cvae_adv_only": "adv_only",
        "fair_cvae_no_adv": "no_adv",
        "fair_cvae_full": "full",
    }
    if model_key in mode_map:
        return run_fair_cvae_mode(
            str(csv_path), mode_map[model_key], run_latent=run_latent, log=log_fn, progress=progress_fn
        )
=======
def run_selected_model(csv_path: Path, model_key: str) -> dict[str, Any]:
    if model_key in {"baseline_mlp", "standalone_adv"}:
        return _run_live_torch_model(csv_path, model_key)

    archived = {row["method"]: row for row in load_method_comparison()}
    cvae_map = {
        "fair_cvae_adv_only": "Fair CVAE adv_only",
        "fair_cvae_no_adv": "Fair CVAE no_adv",
        "fair_cvae_full": "Fair CVAE full",
    }
    method_name = cvae_map.get(model_key)
    if method_name and method_name in archived:
        row = archived[method_name]
        return {
            "mode": "archived",
            "model": method_name,
            "accuracy": row.get("accuracy"),
            "f1": row.get("f1"),
            "min_di": row.get("min_di"),
            "note": "This first web release reuses archived Fair CVAE experiment outputs from try.txt instead of running full CVAE training online.",
        }
>>>>>>> origin/main

    raise ValueError(f"Unsupported model key: {model_key}")


<<<<<<< HEAD
def _run_live_torch_model(
    csv_path: Path,
    model_key: str,
    log: Callable[[str], None],
    progress: Callable[[float, str], None],
) -> dict[str, Any]:
    np.random.seed(42)
    torch.manual_seed(42)

    progress(0.05, "Loading dataset")
    log("Loading dataset using c.py")
    data = load_and_prepare_data(str(csv_path))

    if model_key == "baseline_mlp":
        progress(0.2, "Training baseline MLP")
        log("Training baseline MLP (c.py)")
=======
def _run_live_torch_model(csv_path: Path, model_key: str) -> dict[str, Any]:
    np.random.seed(42)
    torch.manual_seed(42)

    data = load_and_prepare_data(str(csv_path))

    if model_key == "baseline_mlp":
>>>>>>> origin/main
        model = SimpleClassifier(data["input_dim"], hidden_dim=128)
        train_baseline_model(
            model,
            data["X_train"],
            data["y_train"],
            epochs=25,
            batch_size=256,
            lr=0.001,
        )
        fairness, pred = evaluate_model(model, data["X_test"], data["df_test"], model_type="simple")
        label = "Baseline MLP"
    else:
<<<<<<< HEAD
        progress(0.2, "Training standalone adversarial baseline")
        log("Training standalone adversarial baseline (c.py)")
=======
>>>>>>> origin/main
        model = AdversarialDebiasingGRL(
            data["input_dim"], hidden_dim=64, num_groups=data["n_groups"]
        )
        train_adversarial_model_grl(
            model,
            data["X_train"],
            data["y_train"],
            data["g_int_train"],
            data["X_val"],
            data["y_val"],
            data["g_int_val"],
            epochs=30,
            batch_size=256,
            lr=0.001,
<<<<<<< HEAD
            verbose=True,
=======
            verbose=False,
>>>>>>> origin/main
        )
        fairness, pred = evaluate_model(model, data["X_test"], data["df_test"], model_type="adversarial")
        label = "Standalone adversarial baseline (c.py)"

<<<<<<< HEAD
    progress(0.9, "Evaluating")
    acc = float(accuracy_score(data["y_test"].numpy(), pred))
    f1 = float(f1_score(data["y_test"].numpy(), pred))

    progress(0.99, "Finalizing")
    log("Model run completed")
    return {
        "mode": "live_background",
=======
    acc = float(accuracy_score(data["y_test"].numpy(), pred))
    f1 = float(f1_score(data["y_test"].numpy(), pred))

    return {
        "mode": "live",
>>>>>>> origin/main
        "model": label,
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "min_di": round(float(fairness["DI"].min()), 4),
        "fairness": fairness.round(4).to_dict(orient="records"),
<<<<<<< HEAD
        "note": "Real execution completed from c.py pipeline.",
=======
        "note": "Live run completed on uploaded data using reused c.py training/evaluation functions with reduced epochs for demo responsiveness.",
>>>>>>> origin/main
    }

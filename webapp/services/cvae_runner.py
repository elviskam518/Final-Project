from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


class _LogWriter:
    def __init__(self, log_fn: Callable[[str], None]) -> None:
        self.log_fn = log_fn

    def write(self, msg: str) -> int:
        text = msg.strip()
        if text:
            self.log_fn(text)
        return len(msg)

    def flush(self) -> None:
        return


def _load_more_strict_module():
    module_path = ROOT / "more strict .py"
    spec = importlib.util.spec_from_file_location("more_strict_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fair_cvae_mode(
    csv_path: str,
    mode: str,
    run_latent: bool,
    log: Callable[[str], None],
) -> dict[str, Any]:
    if mode not in {"adv_only", "no_adv", "full"}:
        raise ValueError(f"Unsupported CVAE mode: {mode}")

    writer = _LogWriter(log)
    with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
        module = _load_more_strict_module()

        torch.manual_seed(42)
        np.random.seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log("Loading data with more strict .py")
        data = module.load_and_prepare_data(csv_path)

        log("Training baseline model (required for calibration and latent comparison)")
        base = module.SimpleClassifier(data["input_dim"]).to(device)
        module.train_baseline(base, data, epochs=150, batch_size=256, lr=1e-3, device=device)

        target_rate, _ = module.baseline_val_pred_rate(base, data, device=device, threshold=0.5)
        baseline_eval = module.evaluate_baseline(base, data, device=device)

        log(f"Training Fair CVAE mode={mode}")
        cvae_model = module.FairCVAE_v4(
            x_dim=data["input_dim"],
            n_sensitive=data["n_genders"],
            z_dim=64,
            hidden_dim=256,
            n_sensitive_directions=3,
        ).to(device)

        module.train_fair_cvae_v4(
            cvae_model,
            data,
            epochs=350,
            batch_size=256,
            lr_main=1e-3,
            lr_adv=2e-3,
            adv_steps=5,
            lambda_hsic=50.0,
            lambda_adv=2.0,
            alpha_max=8.0,
            adv_reset_every=40,
            projection_update_every=20,
            device=device,
            verbose=True,
            mode=mode,
        )

        cvae_model.eval()
        X_val = data["X_val"].to(device)
        a_val = data["a_val"].to(device)
        with torch.no_grad():
            out_val = cvae_model(X_val, a_val, use_grl=False, alpha=0, use_projection=True)
            y_prob_val = torch.sigmoid(out_val["y_logit"]).squeeze().cpu().numpy()

        t = module.threshold_for_target_rate(y_prob_val, target_rate)
        cal = module.evaluate_model_at_threshold(cvae_model, data, threshold=t, device=device)

        latent = None
        if run_latent:
            log("Running latent visualisation")
            latent = module.run_latent_visualisation(
                baseline_model=base,
                cvae_model=cvae_model,
                data=data,
                device=device,
                output_dir="latent_vis",
            )

    images = []
    latent_dir = ROOT / "latent_vis"
    if run_latent and latent_dir.exists():
        images = [f"/latent_vis/{p.name}" for p in sorted(latent_dir.glob("*.png"))]

    return {
        "mode": "live_background",
        "model": f"Fair CVAE {mode}",
        "threshold": float(t),
        "target_rate": float(target_rate),
        "accuracy": round(float(cal["accuracy"]), 4),
        "f1": round(float(cal["f1"]), 4),
        "min_di": round(float(cal["fairness"]["DI"].min()), 4),
        "fairness": cal["fairness"].round(4).to_dict(orient="records"),
        "baseline_reference": {
            "accuracy": round(float(baseline_eval["accuracy"]), 4),
            "f1": round(float(baseline_eval["f1"]), 4),
            "min_di": round(float(baseline_eval["fairness"]["DI"].min()), 4),
        },
        "latent_metrics": latent,
        "latent_images": images,
    }

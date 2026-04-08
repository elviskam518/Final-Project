# Fairness-in-Hiring Academic Demo Web App

This repository now includes a practical **local web demo** for your final-year project on bias mitigation in hiring decisions.

## 1) Pre-implementation inspection and reuse decisions

### Repository files inspected and reused directly
- `b.py`: reused for intermediate fairness analysis (`add_proxy_qualified`, `compute_fairness_metrics`, `compute_odds_ratios`).
- `c.py`: reused for live demo model execution for:
  - Baseline MLP
  - Standalone adversarial baseline (GRL)
- `tech_adversarial_grl_summary.csv`: reused for method comparison (real experimental outputs).
- `tech_adversarial_grl_di_comparison.csv`: reused for per-group DI comparison.
- `try.txt`: parsed to surface:
  - Fair CVAE (`adv_only`, `no_adv`, `full`) performance summaries
  - feature-removal baseline summary
  - latent-space metric comparison block

### Explicit non-exposure rule
- `a.py` is **not** included in any website workflow and is not required by users.

## 2) Proposed (and implemented) folder structure

```text
Final-Project/
├── b.py
├── c.py
├── try.txt
├── tech_adversarial_grl_summary.csv
├── tech_adversarial_grl_di_comparison.csv
├── requirements.txt
└── webapp/
    ├── main.py
    ├── uploads/
    ├── services/
    │   ├── demo_runner.py
    │   └── results_loader.py
    ├── static/
    │   ├── style.css
    │   ├── results.js
    │   └── demo.js
    └── templates/
        ├── base.html
        ├── home.html
        ├── results.html
        ├── demo.html
        └── about.html
```

## 3) Implementation plan (phased)

### Phase 1 — Foundation and routing
- Build a FastAPI app with server-side templates and static assets.
- Create required pages: Home, Results, Interactive Demo, About.

### Phase 2 — Results integration from real project outputs
- Load generated CSV result files and parse existing experiment log(s).
- Present method comparison, per-group DI comparison, and latent-space analysis.

### Phase 3 — Interactive pipeline
- Upload CSV.
- Run intermediate analysis using reused `b.py` functions.
- Let user choose model.
- Execute selected model and return final output.

### Phase 4 — Documentation and local runability
- Provide clear setup/run steps, assumptions, known limitations, and reuse mapping.

## 4) Major risks and assumptions

### Risks
- Full Fair CVAE training in-browser workflow is computationally heavy for a first local demo.
- Uploaded CSV must match expected schema (`Gender`, `Race`, `Hired`, and relevant feature columns) for full compatibility.

### Assumptions in this first version
- **Live execution** is implemented for Baseline MLP + standalone adversarial baseline via `c.py` with reduced epochs for responsiveness.
- Fair CVAE variants are represented using **archived real outputs** parsed from `try.txt` (clearly labeled as archived in API response).
- This still satisfies an end-to-end real-model demo while preserving existing research code and avoiding core logic rewrites.

---

## Website overview

### Pages
- **Home**: concise project summary and supported model families.
- **Results**:
  - baseline bias diagnosis and method comparison
  - per-group DI comparison chart/table
  - latent-space summary table
- **Interactive Demo**:
  1. upload CSV
  2. run intermediate analysis (fairness + odds)
  3. choose model
  4. run selected model
  5. view final output clearly
- **About**: compact academic context and constraints

## Model options represented in UI
- Baseline MLP (live)
- Standalone adversarial baseline from `c.py` (live)
- Fair CVAE `adv_only` (archived result)
- Fair CVAE `no_adv` (archived result)
- Fair CVAE `full` (archived result)

## Backend API

- `GET /api/results`
  - returns method comparison, per-group DI table, latent-space summary.

- `POST /api/demo/analyze`
  - multipart form: `file`, `top_quantile`
  - saves upload and returns intermediate analysis outputs.

- `POST /api/demo/run-model`
  - form: `upload_id`, `model_key`
  - runs selected model (live for baseline/adversarial, archived for Fair CVAE modes).

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run locally

```bash
uvicorn webapp.main:app --reload
```

Open:
- http://127.0.0.1:8000/

## Input file expectation for demo page

For best compatibility, uploaded CSV should include:
- demographic columns: `Gender`, `Race`
- label column: `Hired`
- project features used by `c.py` pipeline (if available):
  - `YearsExperience`, `EducationLevel`, `AlgorithmSkill`, `SystemDesignSkill`,
  - `OverallInterviewScore`, `GitHubScore`, `NumLanguages`, `HasReferral`,
  - `ResumeScore`, `TechInterviewScore`, `CultureFitScore`

## Known limitations

1. Fair CVAE variants are shown from archived experiment outputs in this first web release (not retrained online).
2. No dependency on `a.py` is exposed to users by design.
3. Live training endpoints are reduced-epoch versions for practical local responsiveness.

## Notes on academic wording and scope

- UI is intentionally minimal and presentation-friendly.
- No standalone Methodology or Limitations pages were added; key context is integrated into Home/About/README as requested.

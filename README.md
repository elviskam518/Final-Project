# Fairness-in-Hiring Academic Demo Web App

This repository includes a local web application for your final-year project on bias mitigation in hiring.

## Integrity rules implemented

- The web app uses only real project code paths.
- `a.py` is excluded from end-user workflow.
- No `.txt` log parsing is used to simulate model outputs.
- No fake/archival substitute is used for live demo execution.
- If a model is too expensive for interactive execution, the UI explicitly marks it as disabled and explains why.

## Reused project code

- `b.py`:
  - intermediate fairness diagnostics (`add_proxy_qualified`, `compute_fairness_metrics`, `compute_odds_ratios`)
- `c.py`:
  - live model demo execution for Baseline MLP and standalone adversarial baseline
- `more strict .py`:
  - offline Fair CVAE experiment pipeline
- `latent_vis.py`:
  - latent-space visual outputs (figures written to `latent_vis/`)

## Folder structure

```text
Final-Project/
├── b.py
├── c.py
├── more strict .py
├── latent_vis.py
├── tech_adversarial_grl_summary.csv
├── tech_adversarial_grl_di_comparison.csv
├── requirements.txt
└── webapp/
    ├── main.py
    ├── run_offline_pipeline.py
    ├── uploads/
    ├── offline_results/
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

## Website behavior

### Pages
- Home
- Results
- Interactive Demo
- About

### Interactive Demo workflow
1. Upload CSV.
2. Run intermediate analysis from `b.py`.
3. Choose a model.
4. Run model.
5. View final outputs.

### Model availability in demo
- Baseline MLP (`c.py`): **live enabled**
- Standalone adversarial baseline (`c.py`): **live enabled**
- Fair CVAE `adv_only` / `no_adv` / `full` (`more strict .py`): **disabled in interactive request/response** with explicit message (computationally expensive)

## Results page data sources

- Method comparison and DI tables from generated outputs (e.g., `c.py` CSV outputs).
- Fair CVAE results shown only if generated offline via real `more strict .py` pipeline and saved to `webapp/offline_results/fair_cvae_results.json`.
- Latent-space figures are loaded from `latent_vis/*.png` when present.

## Setup

```bash
pip install -r requirements.txt
```

## Run web app

```bash
uvicorn webapp.main:app --reload
```

Open:
- http://127.0.0.1:8000/

## Generate offline Fair CVAE outputs (real code path)

> This is intentionally offline due runtime cost.

```bash
python webapp/run_offline_pipeline.py
```

This executes `run_experiment()` from `more strict .py` and writes:
- `webapp/offline_results/fair_cvae_results.json`

If `more strict .py` runs latent visualization, latent images are written under:
- `latent_vis/`

## Input CSV requirements

Minimum required columns for intermediate analysis:
- `Gender`, `Race`, `Hired`

For `c.py` live model paths, include project feature columns when available:
- `YearsExperience`, `EducationLevel`, `AlgorithmSkill`, `SystemDesignSkill`,
- `OverallInterviewScore`, `GitHubScore`, `NumLanguages`, `HasReferral`,
- `ResumeScore`, `TechInterviewScore`, `CultureFitScore`

## Known limitations

1. Fair CVAE is intentionally disabled in interactive endpoint due heavy runtime from the real `more strict .py` pipeline.
2. Fair CVAE appears on Results only after running the offline pipeline from real code.
3. This app does not expose `a.py`.

# Fairness-in-Hiring Academic Demo Web App

This repository includes a local web application for your final-year project on bias mitigation in hiring.

## Integrity and execution rules

- The web app uses real project scripts as source of truth:
  - `b.py`
  - `c.py`
  - `more strict .py`
  - `latent_vis.py`
- `a.py` is excluded from end-user workflow.
- No txt-based, archived, placeholder, or fake substitute outputs are used for demo runs.

## Architecture for long-running execution

The app implements a background job flow for real model execution:

1. Upload CSV.
2. Run intermediate analysis from `b.py`.
3. Start a model job via API (`/api/jobs`).
4. Backend executes the real selected pipeline in a background thread.
5. Frontend polls status/logs and displays incremental execution logs.
6. Final result is available at `/api/jobs/{job_id}/result` only when completed.

### Job states
- `queued`
- `running`
- `completed`
- `failed`

## Reused scripts and how they are used

- `b.py`
  - intermediate fairness diagnostics
- `c.py`
  - Baseline MLP and standalone adversarial baseline execution
- `more strict .py`
  - Fair CVAE single-mode execution (`adv_only`, `no_adv`, `full`) in job worker
- `latent_vis.py`
  - latent visualisation invoked after Fair CVAE training when requested

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
    ├── uploads/
    ├── services/
    │   ├── cvae_runner.py
    │   ├── demo_runner.py
    │   ├── job_manager.py
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

## Supported selectable modes (one mode per job)

- Baseline MLP
- Standalone adversarial baseline
- Fair CVAE `adv_only`
- Fair CVAE `no_adv`
- Fair CVAE `full`

## Setup

```bash
pip install -r requirements.txt
```

## Run app

```bash
uvicorn webapp.main:app --reload
```

Open:
- http://127.0.0.1:8000/

## API overview

- `POST /api/demo/analyze`
  - upload CSV and run intermediate diagnostics
- `POST /api/jobs`
  - create long-running model job
- `GET /api/jobs/{job_id}`
  - job status
- `GET /api/jobs/{job_id}/logs?since=N`
  - incremental logs
- `GET /api/jobs/{job_id}/result`
  - final output when completed

## Input CSV requirements

Minimum required columns for intermediate analysis:
- `Gender`, `Race`, `Hired`

Model features expected by project pipelines:
- `YearsExperience`, `EducationLevel`, `AlgorithmSkill`, `SystemDesignSkill`,
- `OverallInterviewScore`, `GitHubScore`, `NumLanguages`, `HasReferral`,
- `ResumeScore`, `TechInterviewScore`, `CultureFitScore`

## Notes / current limitations

- Fair CVAE jobs are computationally expensive and may take significant time.
- Current job system is in-process memory (job history resets on server restart).
- If further scaling is needed, migrate to a persistent queue (e.g., Redis/Celery/RQ) while keeping the same APIs.

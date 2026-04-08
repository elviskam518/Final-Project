from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .services.demo_runner import get_upload_path, run_intermediate_analysis, run_selected_model, save_upload
from .services.job_manager import job_manager
from .services.results_loader import load_latent_summary, load_method_comparison, load_per_group_di_comparison


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
app = FastAPI(title="Fairness in Hiring Demo")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
if (ROOT_DIR / "latent_vis").exists():
    app.mount("/latent_vis", StaticFiles(directory=ROOT_DIR / "latent_vis"), name="latent_vis")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/results", response_class=HTMLResponse)
def results_page(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})


@app.get("/demo", response_class=HTMLResponse)
def demo_page(request: Request):
    return templates.TemplateResponse("demo.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/api/results")
def results_api():
    return {
        "method_comparison": load_method_comparison(),
        "per_group_di": load_per_group_di_comparison(),
        "latent": load_latent_summary(),
    }


@app.post("/api/demo/analyze")
async def analyze_upload(file: UploadFile = File(...), top_quantile: float = Form(0.40)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    data = await file.read()
    upload_id = save_upload(data, file.filename)

    try:
        analysis = run_intermediate_analysis(get_upload_path(upload_id), top_quantile=top_quantile)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"upload_id": upload_id, "analysis": analysis}


@app.post("/api/jobs")
def create_job(
    upload_id: str = Form(...),
    model_key: str = Form(...),
    run_latent: bool = Form(False),
):
    try:
        csv_path = get_upload_path(upload_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    def worker(log):
        log("Job accepted")
        return run_selected_model(csv_path=csv_path, model_key=model_key, run_latent=run_latent, log=log)

    job = job_manager.create_job(model_key=model_key, upload_id=upload_id, worker=worker)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    try:
        job = job_manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    return {
        "job_id": job.job_id,
        "status": job.status,
        "model_key": job.model_key,
        "error": job.error,
        "updated_at": job.updated_at,
        "log_count": len(job.logs),
    }


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str, since: int = 0):
    try:
        job = job_manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    return {
        "job_id": job.job_id,
        "status": job.status,
        "from": since,
        "next": len(job.logs),
        "logs": job.logs[since:],
    }


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    try:
        job = job_manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job status is '{job.status}', result not ready")
    return {"job_id": job.job_id, "result": job.result}

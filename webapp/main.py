from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .services.demo_runner import get_upload_path, run_intermediate_analysis, run_selected_model, save_upload
from .services.results_loader import (
    load_latent_summary,
    load_method_comparison,
    load_per_group_di_comparison,
)


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Fairness in Hiring Demo")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
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
async def analyze_upload(
    file: UploadFile = File(...),
    top_quantile: float = Form(0.40),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    data = await file.read()
    upload_id = save_upload(data, file.filename)

    try:
        analysis = run_intermediate_analysis(get_upload_path(upload_id), top_quantile=top_quantile)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"upload_id": upload_id, "analysis": analysis}


@app.post("/api/demo/run-model")
def run_model(upload_id: str = Form(...), model_key: str = Form(...)):
    try:
        result = run_selected_model(get_upload_path(upload_id), model_key)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


JobStatus = str


@dataclass
class Job:
    job_id: str
    model_key: str
    upload_id: str
    status: JobStatus = "queued"
    progress: float = 0.0
    progress_text: str = "Queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        model_key: str,
        upload_id: str,
        worker: Callable[[Callable[[str], None], Callable[[float, str], None]], dict[str, Any]],
    ) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, model_key=model_key, upload_id=upload_id)
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job_id, worker), daemon=True)
        thread.start()
        return job

    def _run_job(
        self,
        job_id: str,
        worker: Callable[[Callable[[str], None], Callable[[float, str], None]], dict[str, Any]],
    ) -> None:
        self._set_status(job_id, "running")
        self.update_progress(job_id, 0.01, "Job started")

        def log(message: str) -> None:
            self.append_log(job_id, message)

        def progress(value: float, text: str = "") -> None:
            self.update_progress(job_id, value, text)

        try:
            result = worker(log, progress)
            with self._lock:
                job = self._jobs[job_id]
                job.result = result
                job.status = "completed"
                job.progress = 1.0
                job.progress_text = "Completed"
                job.updated_at = time.time()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                job.updated_at = time.time()
                job.logs.append(f"[ERROR] {exc}")

    def _set_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.updated_at = time.time()

    def update_progress(self, job_id: str, value: float, text: str = "") -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.progress = max(0.0, min(1.0, float(value)))
            if text:
                job.progress_text = text
            job.updated_at = time.time()

    def append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.logs.append(message.rstrip())
            job.updated_at = time.time()

    def get_job(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]


job_manager = JobManager()

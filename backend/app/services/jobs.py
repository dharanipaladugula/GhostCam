import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from threading import Lock


@dataclass
class Job:
    job_id: str
    original_filename: str
    upload_path: str
    output_path: Optional[str] = None
    status: str = "uploaded"  # uploaded, processing, completed, failed
    processor: str = "IdentityProcessor"
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobStore:
    """Lightweight in-memory job registry for managing processing jobs."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = Lock()

    def create(self, original_filename: str, upload_path: str) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            original_filename=original_filename,
            upload_path=upload_path,
            status="uploaded",
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_status(
        self,
        job_id: str,
        status: str,
        output_path: Optional[str] = None,
        processor: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            job.status = status
            if output_path is not None:
                job.output_path = output_path
            if processor is not None:
                job.processor = processor
            if error is not None:
                job.error = error
            if result is not None:
                job.result = result
            if status in ("completed", "failed"):
                job.completed_at = time.time()
            return job


job_store = JobStore()

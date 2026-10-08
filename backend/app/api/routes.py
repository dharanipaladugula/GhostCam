from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from app.core.config import UPLOAD_DIR, OUTPUT_DIR, ALLOWED_MIME_TYPES, ALLOWED_VIDEO_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS
from app.services.jobs import job_store
from app.pipeline.engine import VideoPipeline, PipelineExecutionError
from app.pipeline.processor import IdentityProcessor

router = APIRouter()


@router.post("/upload")
async def upload_media(file: UploadFile = File(...)):
    """Uploads a video or image file, creates a unique job, and saves to storage."""
    # Check mime type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported media type: '{file.content_type}'. Allowed types: {list(ALLOWED_MIME_TYPES)}"
        )

    # Sanitize base filename and check extension
    safe_filename = Path(file.filename).name
    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS and ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'."
        )

    # Create tracked job with UUID
    temp_target = UPLOAD_DIR / safe_filename
    job = job_store.create(original_filename=safe_filename, upload_path=str(temp_target))

    # Unique physical filename: {job_id}_{safe_filename}
    unique_filename = f"{job.job_id}_{safe_filename}"
    file_path = UPLOAD_DIR / unique_filename

    # Save uploaded file safely
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    # Update job with final storage path
    job.upload_path = str(file_path)

    return {
        "job_id": job.job_id,
        "message": "Media uploaded successfully",
        "filename": safe_filename,
        "media_type": file.content_type,
        "path": str(file_path),
        "status": job.status,
    }


@router.post("/process/{job_id}")
def process_video_job(job_id: str):
    """Executes the video pipeline on an uploaded video job using the pluggable processor."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    input_path = Path(job.upload_path)
    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Uploaded video file for job '{job_id}' is missing from disk: {input_path}"
        )

    output_path = OUTPUT_DIR / f"protected_{job_id}.mp4"

    job_store.update_status(job_id=job_id, status="processing")

    pipeline = VideoPipeline(processor=IdentityProcessor())

    try:
        result = pipeline.process(
            input_path=input_path,
            output_path=output_path,
        )
        job_store.update_status(
            job_id=job_id,
            status="completed",
            output_path=str(output_path),
            processor=pipeline.processor.__class__.__name__,
            result=result,
        )
        return {
            "job_id": job_id,
            "status": "completed",
            "result": result,
        }

    except PipelineExecutionError as e:
        error_msg = f"{e.args[0]}"
        job_store.update_status(
            job_id=job_id,
            status="failed",
            error=error_msg,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Pipeline execution failed",
                "message": error_msg,
                "frame_index": e.frame_index,
                "details": e.details,
            }
        )
    except Exception as e:
        error_msg = f"Unexpected processing error: {str(e)}"
        job_store.update_status(
            job_id=job_id,
            status="failed",
            error=error_msg,
        )
        raise HTTPException(status_code=500, detail={"error": error_msg})


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Retrieves status and results for a processing job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()


@router.get("/jobs/{job_id}/download")
def download_processed_video(job_id: str):
    """Downloads the protected video output of a completed job."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.status != "completed" or not job.output_path:
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' has not produced an output video yet (status: '{job.status}')."
        )

    output_path = Path(job.output_path)
    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Output video file not found on disk: {output_path}"
        )

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=output_path.name
    )

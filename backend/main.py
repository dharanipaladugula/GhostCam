import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend directory is on sys.path for robust module resolution
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.api.routes import router as api_router

app = FastAPI(
    title="GhostCam Backend",
    description="Privacy-Preserving Video Processing & Evaluation Framework",
    version="0.1.0"
)

# Enable CORS for local development and future UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "GhostCam backend is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# Mount core pipeline routes (/upload, /process/{job_id}, /jobs/{job_id}, /jobs/{job_id}/download)
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
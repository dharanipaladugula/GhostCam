from pathlib import Path

# Paths
# config.py is in backend/app/core/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"

# Ensure runtime directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Media Constraints
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "video/mp4",
    "video/avi",
    "video/x-msvideo",
    "video/quicktime",
    "video/webm",
    "video/x-matroska",
}

# Video Encoder defaults
DEFAULT_CONTAINER = ".mp4"
DEFAULT_CODEC = "mp4v"
FALLBACK_CODECS = ["mp4v", "XVID", "avc1"]

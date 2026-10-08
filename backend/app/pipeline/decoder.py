from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple
import cv2
import numpy as np


class VideoDecodingError(Exception):
    """Raised when video decoding fails or encountered a corrupted frame."""
    def __init__(self, message: str, frame_index: int | None = None):
        super().__init__(message)
        self.frame_index = frame_index


@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


class VideoDecoder:
    """Decodes video files frame-by-frame and extracts stream metadata."""

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")

        self.cap: cv2.VideoCapture | None = None
        self.metadata: VideoMetadata | None = None

    def __enter__(self) -> "VideoDecoder":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> VideoMetadata:
        """Opens video capture and extracts metadata."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise VideoDecodingError(f"Failed to open video file: {self.video_path}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # standard fallback if container lacks FPS tag

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise VideoDecodingError(
                f"Invalid video dimensions ({width}x{height}) in {self.video_path}"
            )

        duration = (frame_count / fps) if (frame_count > 0 and fps > 0) else 0.0

        self.metadata = VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_seconds=duration,
        )
        return self.metadata

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields (frame_idx, frame) tuples sequentially.
        Fails fast if any frame is corrupted or unreadable.
        """
        if self.cap is None or not self.cap.isOpened():
            raise VideoDecodingError("VideoDecoder is not open. Call open() or use context manager.")

        expected_count = self.metadata.frame_count if self.metadata else 0
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                # If we could not read any frame at all
                if frame_idx == 0:
                    raise VideoDecodingError(
                        f"Failed to read the first frame (index 0) from {self.video_path}",
                        frame_index=0
                    )
                # If reading stopped well before expected frame count
                if expected_count > 0 and (expected_count - frame_idx) > 2:
                    raise VideoDecodingError(
                        f"Premature stream end or corrupted frame at index {frame_idx} "
                        f"(expected {expected_count} frames)",
                        frame_index=frame_idx
                    )
                # Normal EOF reached
                break

            if frame is None or frame.size == 0:
                raise VideoDecodingError(
                    f"Corrupted or empty frame encountered at index {frame_idx}",
                    frame_index=frame_idx
                )

            if frame.shape[0] != self.metadata.height or frame.shape[1] != self.metadata.width:
                raise VideoDecodingError(
                    f"Frame dimension mismatch at index {frame_idx}: "
                    f"expected {self.metadata.width}x{self.metadata.height}, "
                    f"got {frame.shape[1]}x{frame.shape[0]}",
                    frame_index=frame_idx
                )

            yield frame_idx, frame
            frame_idx += 1

    def close(self) -> None:
        """Releases the underlying VideoCapture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

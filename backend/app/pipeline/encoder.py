from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np


class VideoEncodingError(Exception):
    """Raised when video encoding or post-encode verification fails."""
    pass


class VideoEncoder:
    """
    Encodes processed video frames to disk with runtime codec negotiation
    and automated post-encode verification.
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        preferred_codec: str = "mp4v",
        fallback_codecs: List[str] | None = None,
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.preferred_codec = preferred_codec
        self.fallback_codecs = fallback_codecs or ["mp4v", "XVID", "avc1"]

        self.writer: cv2.VideoWriter | None = None
        self.active_codec: str | None = None
        self.frames_written: int = 0

    def __enter__(self) -> "VideoEncoder":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        """Initializes VideoWriter with runtime codec availability checks."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        candidate_codecs = [self.preferred_codec] + [
            c for c in self.fallback_codecs if c != self.preferred_codec
        ]

        writer = None
        chosen_codec = None

        for codec in candidate_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_writer = cv2.VideoWriter(
                    str(self.output_path),
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                    isColor=True,
                )
                if test_writer.isOpened():
                    writer = test_writer
                    chosen_codec = codec
                    break
                else:
                    test_writer.release()
            except Exception:
                continue

        if writer is None or not writer.isOpened():
            raise VideoEncodingError(
                f"Could not open VideoWriter for {self.output_path} with any candidate codec: {candidate_codecs}"
            )

        self.writer = writer
        self.active_codec = chosen_codec
        self.frames_written = 0

    def write_frame(self, frame: np.ndarray, frame_idx: int) -> None:
        """Writes a processed frame to the video stream."""
        if self.writer is None or not self.writer.isOpened():
            raise VideoEncodingError("VideoEncoder is not open. Call open() first.")

        if frame is None:
            raise VideoEncodingError(f"Attempted to write None frame at index {frame_idx}")

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise VideoEncodingError(
                f"Cannot encode frame {frame_idx}: size mismatch "
                f"expected ({self.width}x{self.height}), got ({frame.shape[1]}x{frame.shape[0]})"
            )

        self.writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        """Releases the VideoWriter resource."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def verify_output(self) -> Dict[str, Any]:
        """
        Verifies that the generated video file exists, is non-empty,
        and can be successfully opened and decoded by OpenCV.
        """
        if not self.output_path.exists():
            raise VideoEncodingError(f"Output video file was not created: {self.output_path}")

        file_size = self.output_path.stat().st_size
        if file_size == 0:
            raise VideoEncodingError(f"Output video file is 0 bytes: {self.output_path}")

        cap = cv2.VideoCapture(str(self.output_path))
        if not cap.isOpened():
            raise VideoEncodingError(
                f"Encoded output video exists ({file_size} bytes) but cannot be opened by VideoCapture: {self.output_path}"
            )

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise VideoEncodingError(
                f"Encoded output video opened, but first frame could not be decoded: {self.output_path}"
            )

        out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = float(cap.get(cv2.CAP_PROP_FPS))
        out_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return {
            "path": str(self.output_path),
            "size_bytes": file_size,
            "codec": self.active_codec,
            "width": out_width,
            "height": out_height,
            "fps": out_fps,
            "frame_count": out_count,
            "frames_written": self.frames_written,
        }

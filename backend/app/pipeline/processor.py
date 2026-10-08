from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class FrameProcessingError(Exception):
    """Raised when an error occurs during frame processing."""
    def __init__(self, message: str, frame_index: int | None = None):
        super().__init__(message)
        self.frame_index = frame_index


class BaseFrameProcessor(ABC):
    """
    Abstract pluggable frame processor interface.
    Future anonymization techniques (blur, pixelation, masking, neural models)
    will implement this contract.
    """

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Process a single video frame.

        Args:
            frame: Input BGR frame as numpy uint8 array (height, width, channels).
            frame_idx: 0-indexed sequential position of the frame.
            metadata: Video stream properties (fps, dimensions, etc.).

        Returns:
            Processed BGR frame (must preserve shape and uint8 dtype).
        """
        pass


class IdentityProcessor(BaseFrameProcessor):
    """
    Pass-through processor: preserves frames verbatim without altering any pixels.
    Used for verifying end-to-end video decoding, pipeline flow, and encoding.
    """

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        if frame is None:
            raise FrameProcessingError(
                f"Received None frame at index {frame_idx}",
                frame_index=frame_idx
            )
        # Return a copy to ensure caller immutability while preserving exact pixel values
        return frame.copy()

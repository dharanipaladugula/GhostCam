import time
from pathlib import Path
from typing import Dict, Any
from app.pipeline.decoder import VideoDecoder, VideoDecodingError
from app.pipeline.encoder import VideoEncoder, VideoEncodingError
from app.pipeline.processor import BaseFrameProcessor, IdentityProcessor, FrameProcessingError


class PipelineExecutionError(Exception):
    """Raised when the video processing pipeline fails."""
    def __init__(self, message: str, frame_index: int | None = None, details: Dict[str, Any] | None = None):
        super().__init__(message)
        self.frame_index = frame_index
        self.details = details or {}


class VideoPipeline:
    """
    Coordinates end-to-end video processing:
    Decode -> Process -> Encode -> Verify & Compare.
    """

    def __init__(self, processor: BaseFrameProcessor | None = None):
        self.processor = processor or IdentityProcessor()

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        preferred_codec: str = "mp4v"
    ) -> Dict[str, Any]:
        input_path = Path(input_path)
        output_path = Path(output_path)

        start_time = time.time()
        decoder = VideoDecoder(input_path)
        encoder = None

        processed_count = 0
        current_frame_idx = 0

        try:
            with decoder:
                meta = decoder.metadata
                encoder = VideoEncoder(
                    output_path=output_path,
                    fps=meta.fps,
                    width=meta.width,
                    height=meta.height,
                    preferred_codec=preferred_codec,
                )

                with encoder:
                    meta_dict = {
                        "fps": meta.fps,
                        "width": meta.width,
                        "height": meta.height,
                        "expected_frame_count": meta.frame_count,
                    }

                    for frame_idx, frame in decoder.frames():
                        current_frame_idx = frame_idx
                        # Pass through pluggable processor
                        processed_frame = self.processor.process_frame(
                            frame=frame,
                            frame_idx=frame_idx,
                            metadata=meta_dict,
                        )

                        # Write to output encoder
                        encoder.write_frame(processed_frame, frame_idx=frame_idx)
                        processed_count += 1

                # Post-encode verification
                output_verification = encoder.verify_output()

            elapsed_seconds = round(time.time() - start_time, 3)

            input_metrics = {
                "fps": meta.fps,
                "width": meta.width,
                "height": meta.height,
                "frame_count": meta.frame_count,
                "duration_seconds": round(meta.duration_seconds, 3),
            }

            output_metrics = {
                "fps": output_verification["fps"],
                "width": output_verification["width"],
                "height": output_verification["height"],
                "frame_count": output_verification["frame_count"],
                "duration_seconds": round(
                    output_verification["frame_count"] / output_verification["fps"], 3
                ) if output_verification["fps"] > 0 else 0.0,
                "frames_written": output_verification["frames_written"],
                "codec": output_verification["codec"],
                "size_bytes": output_verification["size_bytes"],
            }

            property_preservation = {
                "fps_preserved": abs(input_metrics["fps"] - output_metrics["fps"]) < 0.1,
                "dimensions_preserved": (
                    input_metrics["width"] == output_metrics["width"]
                    and input_metrics["height"] == output_metrics["height"]
                ),
                "frame_count_preserved": (
                    processed_count == output_verification["frames_written"]
                ),
            }

            return {
                "status": "completed",
                "processor": self.processor.__class__.__name__,
                "elapsed_seconds": elapsed_seconds,
                "input_video": {
                    "path": str(input_path),
                    "metrics": input_metrics,
                },
                "output_video": {
                    "path": str(output_path),
                    "metrics": output_metrics,
                },
                "preservation_check": property_preservation,
            }

        except VideoDecodingError as e:
            raise PipelineExecutionError(
                message=f"Decoding failure at frame {e.frame_index}: {str(e)}",
                frame_index=e.frame_index,
                details={"stage": "decoder", "input_file": str(input_path)}
            ) from e
        except FrameProcessingError as e:
            raise PipelineExecutionError(
                message=f"Processing failure at frame {e.frame_index}: {str(e)}",
                frame_index=e.frame_index,
                details={"stage": "processor", "input_file": str(input_path)}
            ) from e
        except VideoEncodingError as e:
            raise PipelineExecutionError(
                message=f"Encoding failure: {str(e)}",
                frame_index=current_frame_idx,
                details={"stage": "encoder", "output_file": str(output_path)}
            ) from e
        except Exception as e:
            raise PipelineExecutionError(
                message=f"Unexpected pipeline error: {str(e)}",
                frame_index=current_frame_idx,
                details={"stage": "pipeline"}
            ) from e

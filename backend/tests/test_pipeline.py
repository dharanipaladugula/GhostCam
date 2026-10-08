"""
End-to-end verification script for GhostCam Video Processing Pipeline.
Tests:
1. FastAPI /, /health endpoints
2. IdentityProcessor pixel fidelity
3. VideoPipeline on synthetic video
4. VideoPipeline on real iLIDS-VID sequence
5. Error handling on corrupt/invalid input
6. FastAPI /upload, /process, /jobs, /jobs/download workflow
"""

import io
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2

# Ensure backend root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main
from app.core.config import UPLOAD_DIR, OUTPUT_DIR
from app.pipeline.decoder import VideoDecoder, VideoDecodingError
from app.pipeline.encoder import VideoEncoder, VideoEncodingError
from app.pipeline.processor import IdentityProcessor
from app.pipeline.engine import VideoPipeline, PipelineExecutionError
from app.api.routes import upload_media, process_video_job, get_job_status, download_processed_video
from app.services.jobs import job_store
from starlette.datastructures import UploadFile


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"TEST: {title}")
    print("=" * 60)


def test_basic_endpoints():
    print_section("FastAPI Basic Endpoints (/ and /health)")
    home_res = main.home()
    health_res = main.health_check()
    print("GET /       ->", home_res)
    print("GET /health ->", health_res)
    assert home_res == {"message": "GhostCam backend is running!"}
    assert health_res == {"status": "healthy"}
    print("PASS: Basic endpoints respond correctly.")


def test_identity_processor_fidelity():
    print_section("IdentityProcessor Pixel-Fidelity")
    processor = IdentityProcessor()
    # Create random test frame
    rng = np.random.RandomState(42)
    sample_frame = rng.randint(0, 256, (128, 64, 3), dtype=np.uint8)
    processed = processor.process_frame(sample_frame, frame_idx=0, metadata={})

    assert np.array_equal(sample_frame, processed), "IdentityProcessor altered pixels!"
    assert sample_frame is not processed, "IdentityProcessor must return distinct copy to prevent in-place mutation"
    print("PASS: IdentityProcessor preserves exact pixel values without mutation.")


def test_pipeline_on_synthetic_video():
    print_section("VideoPipeline on Synthetic Video (50 frames, 64x128, 25 FPS)")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        input_path = temp_dir / "synthetic_input.mp4"
        output_path = temp_dir / "synthetic_output.mp4"

        # Generate synthetic video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(input_path), fourcc, 25.0, (64, 128))
        assert writer.isOpened(), "Could not create synthetic video writer"

        for i in range(50):
            frame = np.full((128, 64, 3), fill_value=(i * 5) % 256, dtype=np.uint8)
            cv2.putText(frame, str(i), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(frame)
        writer.release()

        # Run pipeline
        pipeline = VideoPipeline(processor=IdentityProcessor())
        result = pipeline.process(input_path=input_path, output_path=output_path)

        print("Execution Result:")
        print(f"  Status: {result['status']}")
        print(f"  Input:  {result['input_video']['metrics']}")
        print(f"  Output: {result['output_video']['metrics']}")
        print(f"  Preservation: {result['preservation_check']}")

        assert result["status"] == "completed"
        assert result["preservation_check"]["fps_preserved"] is True
        assert result["preservation_check"]["dimensions_preserved"] is True
        assert result["preservation_check"]["frame_count_preserved"] is True
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify output can be reopened and all 50 frames decoded
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened(), "Output video could not be opened by VideoCapture"
        count = 0
        while True:
            ret, f = cap.read()
            if not ret:
                break
            assert f.shape == (128, 64, 3)
            count += 1
        cap.release()
        print(f"PASS: Verified output video successfully reopened; decoded {count} frames.")
        assert count == 50

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pipeline_on_real_ilids_sequence():
    print_section("VideoPipeline on Real iLIDS-VID Sequence (cam1/person001)")
    dataset_seq = Path(r"C:\Users\hp\Downloads\iLIDS-VID\i-LIDS-VID\sequences\cam1\person001")
    if not dataset_seq.exists():
        print("SKIP: iLIDS-VID dataset directory not found, skipping real sequence test.")
        return

    frames = sorted(dataset_seq.glob("*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    print(f"Found {len(frames)} PNG frames from iLIDS-VID {dataset_seq.name}")
    assert len(frames) == 107

    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Create input video from real frames without modifying any dataset files
        input_path = temp_dir / "ilids_person001.mp4"
        output_path = temp_dir / "ilids_person001_protected.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(input_path), fourcc, 25.0, (64, 128))
        assert writer.isOpened()
        for fpath in frames:
            img = cv2.imread(str(fpath))
            writer.write(img)
        writer.release()

        # Run pipeline
        pipeline = VideoPipeline(processor=IdentityProcessor())
        result = pipeline.process(input_path=input_path, output_path=output_path)

        in_m = result["input_video"]["metrics"]
        out_m = result["output_video"]["metrics"]

        print(f"Input Video:  {in_m['width']}x{in_m['height']} @ {in_m['fps']} FPS | {in_m['frame_count']} frames | {in_m['duration_seconds']}s")
        print(f"Output Video: {out_m['width']}x{out_m['height']} @ {out_m['fps']} FPS | {out_m['frame_count']} frames | {out_m['duration_seconds']}s")
        print(f"Codec: {out_m['codec']} | Size: {out_m['size_bytes']} bytes")
        print(f"Preservation: {result['preservation_check']}")

        assert result["status"] == "completed"
        assert out_m["frames_written"] == 107
        assert out_m["width"] == 64
        assert out_m["height"] == 128

        # Decode output completely
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()
        decoded_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            decoded_frames += 1
        cap.release()

        print(f"PASS: Real iLIDS-VID video successfully processed and decoded {decoded_frames}/107 frames.")
        assert decoded_frames == 107

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_corruption_detection():
    print_section("Error Reporting on Corrupted / Invalid Video")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        corrupted_path = temp_dir / "corrupted.mp4"
        # Write corrupted garbage bytes
        with open(corrupted_path, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_HEADER_GARBAGE_BYTES_123456789")

        output_path = temp_dir / "output.mp4"
        pipeline = VideoPipeline(processor=IdentityProcessor())

        failed = False
        try:
            pipeline.process(input_path=corrupted_path, output_path=output_path)
        except PipelineExecutionError as e:
            failed = True
            print("PASS: Caught expected PipelineExecutionError:")
            print(f"  Message: {e}")
            print(f"  Details: {e.details}")

        assert failed, "Pipeline should have failed on corrupted file!"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_api_workflow():
    print_section("FastAPI Full Workflow: Upload -> Process -> Status -> Download")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Create a small valid sample video to upload
        sample_path = temp_dir / "api_test_clip.mp4"
        writer = cv2.VideoWriter(str(sample_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 128))
        for i in range(25):
            writer.write(np.zeros((128, 64, 3), dtype=np.uint8))
        writer.release()

        # 1. Test Upload
        with open(sample_path, "rb") as f:
            content = f.read()

        upload_file = UploadFile(
            file=io.BytesIO(content),
            filename="api_test_clip.mp4",
            headers={"content-type": "video/mp4"}
        )

        upload_response = await upload_media(file=upload_file)
        job_id = upload_response["job_id"]
        saved_path = Path(upload_response["path"])

        print("1. Upload Response:")
        print(f"   job_id: {job_id}")
        print(f"   saved_path: {saved_path}")
        print(f"   status: {upload_response['status']}")
        assert job_id is not None
        assert saved_path.exists()
        assert job_id in saved_path.name, "Physical file name must incorporate UUID job_id"

        # 2. Test Process
        process_response = process_video_job(job_id=job_id)
        print("2. Process Response:")
        print(f"   status: {process_response['status']}")
        print(f"   result summary: {process_response['result']['preservation_check']}")
        assert process_response["status"] == "completed"

        # 3. Test Job Status Lookup
        status_response = get_job_status(job_id=job_id)
        print("3. Job Status Lookup:")
        print(f"   job status: {status_response['status']}")
        print(f"   output_path: {status_response['output_path']}")
        assert status_response["status"] == "completed"
        assert Path(status_response["output_path"]).exists()

        # 4. Test Download
        download_response = download_processed_video(job_id=job_id)
        print("4. Download Response:")
        print(f"   media_type: {download_response.media_type}")
        print(f"   filename: {download_response.filename}")
        assert download_response.media_type == "video/mp4"

        print("PASS: Full FastAPI API workflow verified successfully.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main_suite():
    import asyncio
    print("\n" + "=" * 60)
    print("RUNNING GHOSTCAM PIPELINE TEST SUITE")
    print("=" * 60)

    test_basic_endpoints()
    test_identity_processor_fidelity()
    test_pipeline_on_synthetic_video()
    test_pipeline_on_real_ilids_sequence()
    test_corruption_detection()
    asyncio.run(test_api_workflow())

    print("\n" + "=" * 60)
    print("ALL ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main_suite()

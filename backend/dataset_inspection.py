from pathlib import Path
import cv2


# Location of the iLIDS-VID sequences folder
DATASET_DIR = Path(
    r"C:\Users\hp\Downloads\iLIDS-VID\i-LIDS-VID\sequences"
)


def inspect_camera(camera_path):
    print("\n" + "=" * 50)
    print(f"Camera: {camera_path.name}")
    print("=" * 50)

    # Find all PNG frames
    frames = list(camera_path.glob("*.png"))

    print(f"Number of PNG frames: {len(frames)}")

    if not frames:
        print("No PNG frames found.")
        return

    # Sort using the actual frame number
    frames.sort(
        key=lambda frame: int(frame.stem.split("_")[-1])
    )

    # First and last frame
    print(f"First frame: {frames[0].name}")
    print(f"Last frame:  {frames[-1].name}")

    # Read first frame
    first_frame = cv2.imread(str(frames[0]))

    if first_frame is None:
        print("ERROR: Could not read first frame.")
        return

    height, width, channels = first_frame.shape

    print(f"Image dimensions: {width} x {height}")
    print(f"Channels: {channels}")

    # -----------------------------------------
    # CHECK FRAME ORDER
    # -----------------------------------------

    numbers = []

    for frame in frames:
        number = int(frame.stem.split("_")[-1])
        numbers.append(number)

    print(f"First 10 frame numbers: {numbers[:10]}")
    print(f"Last 10 frame numbers:  {numbers[-10:]}")

    ordered = all(
        numbers[i] < numbers[i + 1]
        for i in range(len(numbers) - 1)
    )

    print(f"Frames in increasing order: {ordered}")

    # -----------------------------------------
    # CHECK FOR MISSING FRAMES
    # -----------------------------------------

    missing_frames = []

    for i in range(len(numbers) - 1):

        if numbers[i + 1] != numbers[i] + 1:

            missing_frames.append(
                (numbers[i], numbers[i + 1])
            )

    if missing_frames:

        print(
            f"Missing/non-consecutive frame gaps: "
            f"{len(missing_frames)}"
        )

        print(
            f"Examples: {missing_frames[:5]}"
        )

    else:

        print("No missing frame numbers detected.")

    # -----------------------------------------
    # CHECK WHETHER FRAMES CAN BE READ
    # -----------------------------------------

    unreadable = []

    for frame in frames:

        image = cv2.imread(str(frame))

        if image is None:
            unreadable.append(frame.name)

    print(f"Unreadable frames: {len(unreadable)}")

    if unreadable:

        print("Examples:")

        for name in unreadable[:5]:
            print(f"  {name}")

    # -----------------------------------------
    # CHECK IMAGE SIZE CONSISTENCY
    # -----------------------------------------

    consistent = True

    for frame in frames:

        image = cv2.imread(str(frame))

        if image is not None:

            h, w = image.shape[:2]

            if w != width or h != height:

                consistent = False
                break

    print(
        f"All frames same dimensions: {consistent}"
    )


def main():

    person_id = "person001"

    cam1 = DATASET_DIR / "cam1" / person_id
    cam2 = DATASET_DIR / "cam2" / person_id

    print("GHOSTCAM — iLIDS-VID DATASET INSPECTION")
    print("=" * 50)

    print(f"\nPerson: {person_id}")

    print(f"Cam1 exists: {cam1.exists()}")
    print(f"Cam2 exists: {cam2.exists()}")

    # Inspect Camera 1
    if cam1.exists():
        inspect_camera(cam1)

    # Inspect Camera 2
    if cam2.exists():
        inspect_camera(cam2)


if __name__ == "__main__":
    main()
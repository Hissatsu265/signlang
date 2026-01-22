import cv2
import os
import math

def increase_fps_no_interpolation(input_video_path, target_fps=25):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise ValueError("Không mở được video")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    ratio = target_fps / input_fps

    new_frames = []
    acc = 0.0

    for frame in frames:
        acc += ratio
        repeat = int(math.floor(acc))
        acc -= repeat

        for _ in range(repeat):
            new_frames.append(frame)

    base, ext = os.path.splitext(input_video_path)
    output_path = base + f"_25fps{ext}"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    for f in new_frames:
        out.write(f)

    out.release()

    return output_path
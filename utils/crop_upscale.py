import subprocess
import os

def crop_and_upscale_video(
    input_path,
    output_path,
    size=720,
    keep_audio=True
):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    vf = (
        "crop=min(iw\\,ih):min(iw\\,ih):"
        "(iw-min(iw\\,ih))/2:(ih-min(iw\\,ih))/2,"
        f"scale={size}:{size}:flags=lanczos"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
    ]

    if keep_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-an"]

    cmd.append(output_path)

    subprocess.run(cmd, check=True)


# # ví dụ dùng
# if __name__ == "__main__":
#     crop_and_upscale_video(
#         input_path="input.mp4",
#         output_path="output_720.mp4",
#         size=720,
#         keep_audio=True
#     )
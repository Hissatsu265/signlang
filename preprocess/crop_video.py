import os
import subprocess
import uuid

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

def get_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def process_videos_recursive(root_folder):
    for root, _, files in os.walk(root_folder):
        for name in files:
            if not name.lower().endswith(VIDEO_EXTS):
                continue

            old_path = os.path.join(root, name)

            try:
                duration = get_duration(old_path)
            except Exception:
                print(f"Không đọc được duration: {old_path}")
                continue

            start = 0.6

            if duration > 2.5:
                cut_end = 0.8
            elif 2.0 <= duration <= 2.5:
                cut_end = 0.6
            elif 1.5 <= duration <= 1.9:
                cut_end = 0.4
            else:
                cut_end = 0.2

            end = duration - cut_end

            if end <= start:
                print(f"Bỏ qua video quá ngắn: {old_path}")
                continue

            new_name = f"processed_{uuid.uuid4().hex[:8]}.mp4"
            new_path = os.path.join(root, new_name)

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", old_path,
                "-an",
                "-c:v", "libx264",
                "-preset", "fast",
                "-movflags", "+faststart",
                new_path
            ]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            os.remove(old_path)
            print(f"Đã xử lý: {old_path} -> {new_path}")


# ===== sử dụng =====
root_video_folder = "/workspace/signlang/video"
process_videos_recursive(root_video_folder)

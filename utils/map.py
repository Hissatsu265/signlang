import os
import re
import random

BASE_PATH = "/workspace/signlang/video_combine"

def get_video_paths_from_text(text):
    # chuẩn hóa text, bỏ ký tự đặc biệt
    text = text.lower()
    text = re.sub(r"[^a-zA-Zäöüß ]", " ", text)

    words = text.split()
    selected_videos = {}

    for word in words:
        folder_path = os.path.join(BASE_PATH, word)

        if os.path.isdir(folder_path):
            video_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
            ]

            if video_files:
                selected_videos[word] = random.choice(video_files)
            else:
                selected_videos[word] = None
        else:
            selected_videos[word] = None

    return selected_videos


# # ví dụ dùng
# if __name__ == "__main__":
#     text_input = "Ich habe diesen Monat höheres Gehalt"
#     result = get_video_paths_from_text(text_input)

#     for word, path in result.items():
#         print(word, "->", path)

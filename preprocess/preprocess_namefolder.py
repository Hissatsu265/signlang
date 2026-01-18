import os
import re

def rename_folders(parent_dir):
    for name in os.listdir(parent_dir):
        old_path = os.path.join(parent_dir, name)

        if not os.path.isdir(old_path):
            continue

        # bỏ chữ số và dấu - ở đầu
        new_name = re.sub(r'^[0-9\-]+', '', name)

        # strip khoảng trắng + chuyển sang chữ thường
        new_name = new_name.strip().lower()

        if new_name == "" or new_name == name:
            continue

        new_path = os.path.join(parent_dir, new_name)

        # tránh ghi đè
        if os.path.exists(new_path):
            print(f"Bỏ qua vì đã tồn tại: {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Đổi: '{name}' -> '{new_name}'")


# ===== sử dụng =====
root_folder = "/workspace/0signlang"
rename_folders(root_folder)

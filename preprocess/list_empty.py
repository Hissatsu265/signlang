import os

def list_empty_folders(parent_dir):
    empty_folders = []

    for name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, name)

        if not os.path.isdir(folder_path):
            continue

        # nếu bên trong hoàn toàn trống
        if len(os.listdir(folder_path)) == 0:
            empty_folders.append(folder_path)

    return empty_folders


# ===== sử dụng =====
root_folder = "/workspace/signlang/video"
result = list_empty_folders(root_folder)

print("Folder con bị trống hoàn toàn:")
for f in result:
    print(f)
print(f"Tổng cộng: {len(result)} folder trống.")
import os
import shutil
import unicodedata

def normalize_name(name):
    return unicodedata.normalize("NFC", name.strip())

def sync_missing_folders(src_root, dst_root):
    src_folders_raw = os.listdir(src_root)
    dst_folders_raw = os.listdir(dst_root)

    # map: normalized_name -> original_name
    src_map = {normalize_name(name): name for name in src_folders_raw}
    dst_map = {normalize_name(name): name for name in dst_folders_raw}

    missing_keys = set(src_map.keys()) - set(dst_map.keys())

    print(f"Found {len(missing_keys)} missing folders")

    for key in missing_keys:
        folder_name = src_map[key]

        src_path = os.path.join(src_root, folder_name)
        dst_path = os.path.join(dst_root, folder_name)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
            print(f"Copied: {folder_name}")

if __name__ == "__main__":
    folder_1 = "/workspace/signlang/signlang"
    folder_2 = "/workspace/signlang/video_combine"

    sync_missing_folders(folder_1, folder_2)

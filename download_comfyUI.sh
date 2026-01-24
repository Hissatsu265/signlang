# #!/bin/bash
# set -e

# # Lấy thư mục chứa file .sh hiện tại
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# echo "SCRIPT_DIR = $SCRIPT_DIR"

# # Tải dữ liệu
# wget https://huggingface.co/datasets/Hissatsu265/Signlang_ver2/resolve/main/comfyui.zip
# wget https://huggingface.co/datasets/Hissatsu265/Signlang_ver2/resolve/main/pose_json.zip

# # Giải nén
# unzip -o comfyui.zip
# unzip -o pose_json.zip

# # Tạo thư mục signlang nếu chưa có
# mkdir -p "$SCRIPT_DIR"

# # Di chuyển dữ liệu
# mv "$SCRIPT_DIR/pose_json" "$SCRIPT_DIR"
# mv "$SCRIPT_DIR/workspace/signlang/ComfyUI" "$SCRIPT_DIR"
# echo "Setup completed successfully"
# rm -rf "$SCRIPT_DIR/workspace"
# rm -rf "$SCRIPT_DIR/comfyui.zip"
# rm -rf "$SCRIPT_DIR/pose_json.zip"


# pip install -r requirements.txt
# pip install opencv-python sageattention

# # Cài đặt ComfyUI
# cd "$SCRIPT_DIR/ComfyUI"
# pip install -r requirements.txt

# # Custom nodes
# cd "$SCRIPT_DIR/ComfyUI/custom_nodes/comfyui-kjnodes"
# pip install -r requirements.txt

# cd "$SCRIPT_DIR/ComfyUI/custom_nodes/comfyui-videohelpersuite"
# pip install -r requirements.txt

# cd "$SCRIPT_DIR/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper"
# pip install -r requirements.txt
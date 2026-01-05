#!/bin/bash

# Xác định thư mục chứa script này
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Thư mục đích
BASE_DIR="/workspace/signlang"
ZIP_PATH="$BASE_DIR/ComfyUI.zip"

# URL tải
COMFYUI_URL="https://huggingface.co/datasets/Hissatsu265/signlang/resolve/main/ComfyUI.zip"

echo "📁 Base directory: $BASE_DIR"

# Tạo thư mục nếu chưa tồn tại
echo "🔧 Creating directory if not exist..."
mkdir -p "$BASE_DIR"

echo "📥 Downloading ComfyUI package..."

# Tải zip, không tải lại nếu đã tồn tại
wget -nc -O "$ZIP_PATH" "$COMFYUI_URL"

echo "📦 Unzipping ComfyUI..."

# Giải nén, ghi đè nếu đã tồn tại
unzip -o "$ZIP_PATH" -d "$BASE_DIR"

echo "🧹 Cleaning up zip file..."
rm -f "$ZIP_PATH"

echo "✅ ComfyUI downloaded and extracted successfully!"

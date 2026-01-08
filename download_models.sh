#!/bin/bash

# Xác định thư mục chứa script này
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/workspace/signlang/ComfyUI/models"

# Các thư mục con
DIFFUSION_DIR="$BASE_DIR/diffusion_models"
TEXT_ENCODER_DIR="$BASE_DIR/text_encoders"
CLIP_VISION_DIR="$BASE_DIR/clip_vision"
VAE_DIR="$BASE_DIR/vae"
LORA_DIR="$BASE_DIR/loras"
# ======================================================================

# ========================================================================
echo "📁 Base directory: $BASE_DIR"

# Tạo thư mục nếu chưa tồn tại
echo "🔧 Creating directories if not exist..."
mkdir -p "$DIFFUSION_DIR" "$TEXT_ENCODER_DIR" "$CLIP_VISION_DIR" "$VAE_DIR" "$LORA_DIR"

echo "📥 Downloading models..."

# Download Lightx2v LoRA
wget -nc -O "$LORA_DIR/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
"https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

# Download CLIP Vision
wget -nc -O "$CLIP_VISION_DIR/clip_vision_h.safetensors" \
"https://huggingface.co/fofr/comfyui/resolve/main/clip_vision/clip_vision_h.safetensors"

# Download VAE
wget -nc -O "$VAE_DIR/Wan2_1_VAE_bf16.safetensors" \
"https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"

# Download UMT5 Text Encoder
wget -nc -O "$TEXT_ENCODER_DIR/umt5-xxl-enc-bf16.safetensors" \
"https://huggingface.co/Serenak/chilloutmix/resolve/main/umt5-xxl-enc-bf16.safetensors"

# Download SteadyDancer diffusion model
wget -nc -O "$DIFFUSION_DIR/Wan21_SteadyDancer_fp8_e4m3fn_scaled_KJ.safetensors" \
"https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/SteadyDancer/Wan21_SteadyDancer_fp8_e4m3fn_scaled_KJ.safetensors"

echo "✅ All models downloaded successfully!"
# =============================================================
ZIP_PATH="$BASE_DIR/video.zip"

# URL tải
VIDEO_URL="https://huggingface.co/datasets/Hissatsu265/signlang/resolve/main/video.zip"

echo "📁 Base directory: $BASE_DIR"

# Tạo thư mục nếu chưa tồn tại
echo "🔧 Creating directory if not exist..."
mkdir -p "$BASE_DIR"

echo "📥 Downloading sign language dataset..."

# Tải file zip, không tải lại nếu đã tồn tại
wget -nc -O "$ZIP_PATH" "$VIDEO_URL"

echo "📦 Unzipping dataset..."

# Giải nén
unzip -o "$ZIP_PATH" -d "$BASE_DIR"

echo "🧹 Cleaning up zip file..."
rm -f "$ZIP_PATH"

echo "✅ Dataset downloaded and extracted successfully!"
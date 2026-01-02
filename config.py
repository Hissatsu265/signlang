import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Output directory
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Upload directory
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Log directory
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Redis configuration for job queue
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8003))

# Job cleanup configuration
JOB_RETENTION_HOURS = int(os.getenv("JOB_RETENTION_HOURS", 24))  # Giữ job trong 24h
VIDEO_RETENTION_HOURS = int(os.getenv("VIDEO_RETENTION_HOURS", 72))  # Giữ video trong 72h
CLEANUP_INTERVAL_MINUTES = int(os.getenv("CLEANUP_INTERVAL_MINUTES", 180))  # Cleanup mỗi 60 phút

# RunPod serverless mode
RUNPOD_MODE = os.getenv("RUNPOD_MODE", "false")  # Enable serverless mode when "true"

# Performance Configuration
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", 3))  # Max parallel video jobs (auto-calculated based on GPU)
GPU_MEMORY_PER_JOB_GB = int(os.getenv("GPU_MEMORY_PER_JOB_GB", 15))  # GPU memory reserved per job
GPU_MEMORY_RESERVE_GB = int(os.getenv("GPU_MEMORY_RESERVE_GB", 5))  # Safety buffer for system
ENABLE_PARALLEL_PROCESSING = os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"  # Enable/disable parallel processing

# ComfyUI Configuration
COMFYUI_BASE_PORT = int(os.getenv("COMFYUI_BASE_PORT", 8188))  # Starting port for ComfyUI instances (8188, 8189, 8190...)

# Job Time Estimation
AVERAGE_JOB_TIME_MINUTES = int(os.getenv("AVERAGE_JOB_TIME_MINUTES", 25))  # Average time per job

# Job Retry Configuration
MAX_JOB_RETRIES = int(os.getenv("MAX_JOB_RETRIES", 3))  # Number of retry attempts (default: 3)
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", 30))  # Delay between retries in seconds (default: 30s)
USE_EXPONENTIAL_BACKOFF = os.getenv("USE_EXPONENTIAL_BACKOFF", "true").lower() == "true"  # Use exponential backoff for retries

# =======================================
SERVER_COMFYUI="127.0.0.1:8188"
WORKFLOW_INFINITETALK_PATH="/workflow/InfiniteTalk_api_ver2.json"
# =======================================
from dotenv import load_dotenv
import os
load_dotenv()

class DirectusConfig:
    DIRECTUS_URL = os.getenv("DIRECTUS_URL")
    ACCESS_TOKEN = os.getenv("DIRECTUS_ACCESS_TOKEN")
    FOLDER_ID = os.getenv("DIRECTUS_FOLDER_ID")


# =====================================================
# from datetime import datetime
# from pymongo import MongoClient


# pass_db=os.getenv("MONGODB_PASSWORD","MONGODB_Pass")
# MONGODB_URI = os.getenv("MONGODB_URI", "")
# MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "anymateme_eduhub_prod")
# MONGODB_JOBS_COLLECTION = os.getenv("MONGODB_JOBS_COLLECTION", "video_jobs")
# MONGODB_EFFECT_JOBS_COLLECTION = os.getenv("MONGODB_EFFECT_JOBS_COLLECTION", "effect_jobs")
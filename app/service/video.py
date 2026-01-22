import asyncio
import json
import uuid
import time
import os
import glob
import socket
import aiohttp
import websockets
import aiofiles
from pathlib import Path

# Cấu hình
BASE_DIR = Path("/workspace/signlang")
COMFYUI_DIR = BASE_DIR / "ComfyUI"
OUTPUT_DIR = COMFYUI_DIR / "output"
SERVER_ADDRESS = "127.0.0.1:8188"

# ========== 1. Khởi động ComfyUI ==========
async def wait_for_port(host: str, port: int, timeout: int = 60) -> bool:
    """Đợi cho đến khi port được mở"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"✅ ComfyUI port {port} đã sẵn sàng!")
                return True
        except (OSError, ConnectionRefusedError):
            await asyncio.sleep(2)
    print(f"❌ Timeout {timeout}s - ComfyUI chưa mở port {port}")
    return False


async def start_comfyui():
    """Khởi động ComfyUI server"""
    HOST = "127.0.0.1"
    PORT = 8188
    
    process = await asyncio.create_subprocess_exec(
        "python3", "main.py",
        cwd=str(COMFYUI_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    print(f"🚀 ComfyUI started (PID: {process.pid}) - đang chờ port {PORT}...")
    
    ready = await wait_for_port(HOST, PORT, timeout=120)
    
    if not ready:
        print("⚠️ ComfyUI không khởi động đúng cách")
    else:
        print("🎉 ComfyUI sẵn sàng!")
    
    return process


async def stop_comfyui(process):
    """Dừng ComfyUI server"""
    if process and process.returncode is None:
        print("🛑 Stopping ComfyUI...")
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            print("⚠️ Force killing ComfyUI...")
            process.kill()
            await process.wait()


# ========== 2. Load và gửi Workflow ==========
async def load_workflow(workflow_path: str):
    """Load workflow từ file JSON"""
    async with aiofiles.open(workflow_path, "r", encoding='utf-8') as f:
        content = await f.read()
        return json.loads(content)


async def queue_prompt(workflow, server_address=SERVER_ADDRESS):
    """Gửi workflow đến ComfyUI"""
    client_id = str(uuid.uuid4())
    
    payload = {
        "prompt": workflow,
        "client_id": client_id
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{server_address}/prompt",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                result["client_id"] = client_id
                return result
            else:
                raise Exception(f"Failed to queue prompt: {response.status}")


# ========== 3. Theo dõi tiến trình qua WebSocket ==========
async def wait_for_completion(prompt_id, client_id, server_address=SERVER_ADDRESS):
    """Đợi workflow hoàn thành qua WebSocket"""
    websocket_url = f"ws://{server_address}/ws?clientId={client_id}"
    
    try:
        async with websockets.connect(websocket_url) as websocket:
            completed_nodes = 0
            
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        
                        if data["type"] == "execution_start":
                            print(f"🚀 Bắt đầu workflow: {data.get('data', {}).get('prompt_id')}")
                        
                        elif data["type"] == "executing":
                            node_id = data["data"]["node"]
                            current_prompt_id = data.get("data", {}).get("prompt_id")
                            
                            if current_prompt_id == prompt_id:
                                if node_id is None:
                                    print("🎉 Workflow hoàn thành!")
                                    return True
                                else:
                                    completed_nodes += 1
                                    print(f"⚙️  Đang xử lý node: {node_id} ({completed_nodes})")
                        
                        elif data["type"] == "progress":
                            progress_data = data.get("data", {})
                            value = progress_data.get("value", 0)
                            max_value = progress_data.get("max", 100)
                            node = progress_data.get("node")
                            percentage = (value / max_value * 100) if max_value > 0 else 0
                            print(f"📊 Node {node}: {value}/{max_value} ({percentage:.1f}%)")
                        
                        elif data["type"] == "execution_error":
                            print(f"❌ Lỗi: {data}")
                            return False
                        
                        elif data["type"] == "execution_cached":
                            cached_nodes = data.get("data", {}).get("nodes", [])
                            print(f"💾 {len(cached_nodes)} nodes đã cache")
                
                except asyncio.TimeoutError:
                    print("⏰ WebSocket timeout, đang chờ...")
                    continue
                except Exception as e:
                    print(f"❌ Lỗi WebSocket: {e}")
                    break
    
    except Exception as e:
        print(f"❌ Không thể kết nối WebSocket: {e}")
        return False


# ========== 4. Tìm video output ==========
async def find_latest_video(prefix, output_dir=str(OUTPUT_DIR)):
    """Tìm video mới nhất theo prefix"""
    def _find_files():
        patterns = [
            f"{prefix}*.mp4",
            f"{prefix}_00001.mp4"
        ]
        
        all_files = []
        for pattern in patterns:
            files = glob.glob(os.path.join(output_dir, pattern))
            all_files.extend(files)
        
        if not all_files:
            print(f"🔍 Không tìm thấy video với prefix '{prefix}' trong {output_dir}")
            return None
        
        latest_file = max(all_files, key=os.path.getmtime)
        print(f"📁 Tìm thấy file mới nhất: {latest_file}")
        return latest_file
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _find_files)


# ========== 5. Hàm chính - Run workflow ==========
async def run_workflow(workflow_path: str, job_id: str, image_path, video_path):
    comfy_process = await start_comfyui()
    
    try:
        print("🔄 Đang load workflow...")
        workflow = await load_workflow(workflow_path)
        
        workflow["75"]["inputs"]["video"] = video_path
        workflow["76"]["inputs"]["image"] = image_path
        if "135" in workflow and "inputs" in workflow["135"]:
            workflow["135"]["inputs"]["filename_prefix"] = job_id
        
        print("📤 Đang gửi workflow đến ComfyUI...")
        resp = await queue_prompt(workflow)
        prompt_id = resp["prompt_id"]
        client_id = resp["client_id"]
        print(f"✅ Đã gửi workflow! Prompt ID: {prompt_id}")
        
        success = await wait_for_completion(prompt_id, client_id)
        
        if not success:
            print("❌ Workflow thất bại")
            return None
        
        print("🔍 Đang tìm video...")
        video_path = await find_latest_video(job_id)
        
        if video_path:
            file_size = os.path.getsize(video_path)
            print(f"📏 Kích thước file: {file_size / (1024*1024):.2f} MB")
            return video_path
        else:
            print("❌ Không tìm thấy video")
            return None
    
    finally:
        await stop_comfyui(comfy_process)

import os
import time

def wait_for_valid_video(
    file_path,
    min_size_mb=1.0,
    timeout_sec=30,
    check_interval=1.0
):
    """
    Kiểm tra file video có đạt dung lượng tối thiểu hay chưa.
    Nếu sau timeout vẫn chưa đạt thì vẫn return True để dùng tiếp.

    Returns:
        True  -> file dùng được
        False -> file không tồn tại
    """

    min_size_bytes = min_size_mb * 1024 * 1024
    start_time = time.time()

    while True:
        if not os.path.exists(file_path):
            time.sleep(check_interval)
            if time.time() - start_time > timeout_sec:
                return False
            continue

        current_size = os.path.getsize(file_path)

        if current_size >= min_size_bytes:
            return True

        if time.time() - start_time >= timeout_sec:
            # hết thời gian chờ nhưng vẫn cho dùng tiếp
            return True

        time.sleep(check_interval)

# ========== Ví dụ sử dụng ==========
async def convert_posetovideo(video_path_input, image_path=""):
    workflow_path = "/workspace/signlang/workflow/steadydance_signlang_api.json"
    job_id = f"test_{int(time.time())}"
    
    ok = wait_for_valid_video(
        file_path=video_path_input,
        min_size_mb=0.3,   
        timeout_sec=30
    )

    if ok:
        print("Video sẵn sàng để dùng")


    print(f"🔄 Chuyển pose sang video cho: {video_path_input}")
    video_path = await run_workflow(
        workflow_path=workflow_path,
        job_id=job_id,
        video_path=video_path_input,
        image_path="/workspace/signlang/Gemini_Generated_Image_2vy9gn2vy9gn2vy91 (1).JPG"
    )
    
    if video_path:
        print(f"✅ Thành công! Video tại: {video_path}")
    else:
        print("❌ Thất bại!")
    
    return video_path


# if __name__ == "__main__":
#     asyncio.run(main())
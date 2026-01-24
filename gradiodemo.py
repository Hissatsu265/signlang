import gradio as gr
import threading
import time
# from app.service.llm import SignLanguageMapper
from app.service.llm_ver2 import SignLanguageMapper
# from app.service.extract_pose import SignLanguagePoseTransition
from app.service.extract_pose2 import OpenPoseVideoRenderer

from app.service.video import convert_posetovideo
from utils.crop_upscale import crop_and_upscale_video
from utils.map import get_video_paths_from_text
from utils.dgs_structure import dgs_postprocess
from utils.fps import increase_fps_no_interpolation
import asyncio
import os
import tempfile
import subprocess

# Biến global để quản lý trạng thái xử lý
is_processing = False
processing_lock = threading.Lock()
# mapper = None 

def process_text_to_video(text_input):
    """Hàm xử lý text và trả về 2 video và text kết quả"""
    global is_processing, mapper
    
    with processing_lock:
        if is_processing:
            # Trả về: Video1=None, Video2=None, Thông báo hệ thống, Text kết quả=""
            return None, None, "⚠️ System is being used by another user. Please try again later.", ""
        is_processing = True
    
    try:
        if not text_input or text_input.strip() == "":
            return None, None, "❌ Please enter text content.", ""
# ==================LLM===============================================
        
        print(f"Processing: {text_input}")
        # result = mapper.process(text_input) 
        result = None

        temp_mapper = SignLanguageMapper(video_dir="/workspace/signlang/pose_json")
        result = temp_mapper.process(text_input)
        # ================xử lí cấu trúc======================
        with open("result_log.txt", "a", encoding="utf-8") as f:
            f.write(f"==============\nResult: {result}\n")
        result['output'] = dgs_postprocess(result['output'], text_input)
        print(f"Result: {result}")
        with open("result_log.txt", "a", encoding="utf-8") as f:
            f.write(f"2Result: {result}\n==============\n")
        # ====================================================
        def async_cleanup():
            time.sleep(0.1)  # Đợi một chút
            if temp_mapper is not None:
                temp_mapper.cleanup()
        
        cleanup_thread = threading.Thread(target=async_cleanup, daemon=True)
        cleanup_thread.start()
        print("LLM processing complete.")

# =====================Extracpose===========================================
        path_video_worlds = get_video_paths_from_text(result['output'])
        json_paths = []
        video_paths = []

        for word, path in path_video_worlds:
            if path:
                json_paths.append(path)
                video_path = path.replace('pose_json', 'fullvideo').replace('.json', '.mp4')
                if os.path.exists(video_path):
                    video_paths.append(video_path)
                elif path == "/workspace/signlang/ref_pose/nguoithat_00001.json":
                    print(f"Video not found for path: {video_path}, using black video instead.")
                else:
                    default_path = '/workspace/signlang/black_video.mp4'
                    video_paths.append(default_path)
        # ===================================================
        target_fps = 16
        output_path = "/workspace/signlang/output_concat.mp4"

        tmp_dir = tempfile.mkdtemp()
        normalized_videos = []

        for i, vp in enumerate(video_paths):
            out_v = os.path.join(tmp_dir, f"norm_{i}.mp4")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", vp,
                "-r", str(target_fps),
                "-vsync", "cfr",
                out_v
            ], check=True)
            normalized_videos.append(out_v)

        list_file = os.path.join(tmp_dir, "list.txt")
        with open(list_file, "w") as f:
            for v in normalized_videos:
                f.write(f"file '{v}'\n")

        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ], check=True)
        
        # LƯU PATH CỦA VIDEO THỨ NHẤT (concat video)
        concat_video_path = output_path
        
        # ======================================================
        # pose_extractor = SignLanguagePoseTransition()
        pose_extractor= OpenPoseVideoRenderer()
        output_path = "multi_transition_output.mp4"

        transition_frames = 30
        pose_extractor.merge_multi_json_videos(
            json_paths=json_paths,
            output_path=output_path,
            transition_frames=8,
            fps=16,
            width=720,
            height=720
        )

        # =====================Convert to video=============================
        final_video_path = asyncio.run(convert_posetovideo("/workspace/signlang/multi_transition_output.mp4"))
        # ===================================================
        final_video_path = increase_fps_no_interpolation(final_video_path, target_fps=25)
        
        # RETURN 2 VIDEO: concat_video_path và final_video_path
        return concat_video_path, final_video_path, "✅ Processing complete! Videos are ready.", result['output']
        
    except Exception as e:
        return None, None, f"❌ Processing error: {str(e)}", ""
    
    finally:
        with processing_lock:
            is_processing = False

def check_status():
    """Check system status"""
    if is_processing:
        return "🔴 Processing - System busy"
    return "🟢 Ready - System available"

with gr.Blocks(title="Video Processing Demo") as demo:
    gr.Markdown("# 🎬 Video Processing Demo")
    gr.Markdown("Enter text to process and receive videos & text result")
    
    with gr.Row():
        status_display = gr.Textbox(
            label="System Status",
            value=check_status(),
            interactive=False,
            scale=1
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Enter text content",
                placeholder="Enter content you want to process...",
                lines=5
            )
            
            process_btn = gr.Button("🚀 Process", variant="primary", size="lg")
            
            status_output = gr.Textbox(
                label="Notification",
                interactive=False
            )
        
        with gr.Column(scale=1):
            # Ô hiển thị Text trả về
            text_output = gr.Textbox(
                label="Result Text Content",
                interactive=False,
                placeholder="Result text will appear here..."
            )
            
            # VIDEO 1: Concatenated Video
            video_output_1 = gr.Video(
                label="Concatenated Video",
                interactive=False
            )
            
            # VIDEO 2: Final Processed Video
            video_output_2 = gr.Video(
                label="Final Processed Video",
                interactive=False
            )
    
    # Handle events - CẬP NHẬT: Thêm video_output_1 vào danh sách outputs
    process_btn.click(
        fn=process_text_to_video,
        inputs=[text_input],
        outputs=[video_output_1, video_output_2, status_output, text_output]
    ).then(
        fn=check_status,
        outputs=[status_display]
    )
    
    timer = gr.Timer(2) 
    timer.tick(fn=check_status, outputs=status_display)
    
    gr.Markdown("""
    ---
    ### 📝 User Guide:
    1. Check system status (🟢 Ready / 🔴 Processing)
    2. Enter text content in the left box
    3. Click "Process" button to start
    4. View the **Result Text** and **2 Videos** on the right:
       - **Concatenated Video**: Video ghép từ các video gốc
       - **Final Processed Video**: Video sau khi xử lý pose và convert
    """)

if __name__ == "__main__":
    # mapper = SignLanguageMapper(video_dir="/workspace/signlang/video")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft()
    )
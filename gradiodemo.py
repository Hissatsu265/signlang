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
# Biến global để quản lý trạng thái xử lý
is_processing = False
processing_lock = threading.Lock()
# mapper = None 

def process_text_to_video(text_input):
    """Hàm xử lý text và trả về video và text kết quả"""
    global is_processing, mapper
    
    with processing_lock:
        if is_processing:
            # Trả về: Video=None, Thông báo hệ thống, Text kết quả=""
            return None, "⚠️ System is being used by another user. Please try again later.", ""
        is_processing = True
    
    try:
        if not text_input or text_input.strip() == "":
            return None, "❌ Please enter text content.", ""
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
        path_video_worlds= get_video_paths_from_text(result['output'])
        json_paths=[]
        for word, path in path_video_worlds:
            if path:
                json_paths.append(path)
        
        # pose_extractor = SignLanguagePoseTransition()
        pose_extractor= OpenPoseVideoRenderer()
        output_path = "multi_transition_output.mp4"

        transition_frames = 30
        # pose_extractor.create_multi_video_transition(
        #     video_paths=list_path_videos,
        #     output_path=output_path,
        #     transition_frames=transition_frames
        # )
        pose_extractor.merge_multi_json_videos(
            json_paths=json_paths,
            output_path=output_path,
            transition_frames=8,
            fps=16,
            width=720,
            height=720
        )

        # crop_and_upscale_video(
        #     input_path=output_path,
        #     output_path="/workspace/signlang/output_720.mp4",
        #     size=720,
        #     keep_audio=False
        # )

        # =====================Convert to video=============================
        final_video_path = asyncio.run(convert_posetovideo("/workspace/signlang/multi_transition_output.mp4"))
        # ===================================================
        final_video_path = increase_fps_no_interpolation(final_video_path, target_fps=25)
        # video_path = "/workspace/signlang/video/-frei/processed_45750628.mp4"
        return final_video_path, "✅ Processing complete! Video is ready.", result['output']
        
    except Exception as e:
        return None, f"❌ Processing error: {str(e)}", ""
    
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
    gr.Markdown("Enter text to process and receive video & text result")
    
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
            # THÊM: Ô hiển thị Text trả về
            text_output = gr.Textbox(
                label="Result Text Content",
                interactive=False,
                placeholder="Result text will appear here..."
            )
            
            video_output = gr.Video(
                label="Result Video",
                interactive=False
            )
    
    # Handle events - CẬP NHẬT: Thêm text_output vào danh sách outputs
    process_btn.click(
        fn=process_text_to_video,
        inputs=[text_input],
        outputs=[video_output, status_output, text_output]
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
    4. View the **Result Text** and **Video** on the right
    """)

if __name__ == "__main__":
    # mapper = SignLanguageMapper(video_dir="/workspace/signlang/video")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft()
    )
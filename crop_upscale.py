from moviepy.editor import VideoFileClip
import os

def process_video(input_path, output_path):
    try:
        # Tải video
        clip = VideoFileClip(input_path)
        
        # Lấy kích thước gốc
        w, h = clip.size
        print(f"Kích thước gốc: {w}x{h}")

        # Xác định kích thước để crop về 1:1 (lấy theo cạnh ngắn nhất)
        min_dimension = min(w, h)
        
        # Crop từ chính giữa
        # x_center, y_center là tọa độ tâm
        # width, height là kích thước vùng muốn lấy
        cropped_clip = clip.crop(
            width=min_dimension, 
            height=min_dimension, 
            x_center=w/2, 
            y_center=h/2
        )

        # Resize về 720x720
        final_clip = cropped_clip.resize(newsize=(720, 720))

        # Xuất video
        print("Đang xử lý và lưu video...")
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        # Đóng clip để giải phóng bộ nhớ
        clip.close()
        final_clip.close()
        print(f"Hoàn thành! Video đã được lưu tại: {output_path}")

    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    # Nhập đường dẫn từ người dùng
    video_path = "/content/transition_output_fixed.mp4"
    
    if os.path.exists(video_path):
        output_name = "output_720x720.mp4"
        process_video(video_path, output_name)
    else:
        print("Đường dẫn không tồn tại!")
import cv2
import numpy as np
import os
from pathlib import Path
from collections import Counter
import shutil

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("MediaPipe chưa cài đặt. Cài đặt bằng: pip install mediapipe")

class VideoOrientationDetector:
    """Công cụ phát hiện hướng quay của người trong video (front/left/right)"""
    
    def __init__(self):
        if not MP_AVAILABLE:
            raise ImportError("Cần cài MediaPipe: pip install mediapipe")
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_orientation(self, video_path, sample_frames=30, verbose=False):
        """
        Phát hiện hướng quay của người trong video
        
        Args:
            video_path: Đường dẫn đến file video
            sample_frames: Số frames đầu tiên để phân tích (mặc định 30)
            verbose: Hiển thị chi tiết quá trình (mặc định False)
        
        Returns:
            str: 'front' (chính diện), 'left' (quay trái), 'right' (quay phải), 'unknown' (không xác định)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            if verbose:
                print(f"⚠️  Không thể mở video: {video_path}")
            return 'unknown'
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if verbose:
            print(f"📹 Video: {total_frames} frames, {fps:.1f} fps")
            print(f"🔍 Phân tích {min(sample_frames, total_frames)} frames đầu tiên...")
        
        orientations = []
        frame_count = 0
        
        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        ) as pose:
            
            while cap.isOpened() and frame_count < sample_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Xử lý frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    orientation = self._analyze_frame_orientation(landmarks)
                    
                    if orientation:
                        orientations.append(orientation)
                        if verbose:
                            print(f"Frame {frame_count}: {orientation}")
        
        cap.release()
        
        if not orientations:
            if verbose:
                print("⚠️  Không phát hiện được pose trong video")
            return 'unknown'
        
        # Lấy orientation phổ biến nhất
        final_orientation = max(set(orientations), key=orientations.count)
        
        if verbose:
            count = Counter(orientations)
            confidence = count[final_orientation] / len(orientations) * 100
            print(f"📊 Kết quả: Front={count.get('front', 0)}, Left={count.get('left', 0)}, Right={count.get('right', 0)}")
            print(f"✅ Kết luận: {final_orientation.upper()} ({confidence:.1f}% confidence)\n")
        
        return final_orientation
    
    def _analyze_frame_orientation(self, landmarks):
        """
        Phân tích hướng cơ thể từ landmarks của 1 frame
        """
        # Chỉ số landmarks theo MediaPipe Pose
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_EYE = 2
        RIGHT_EYE = 5
        NOSE = 0
        
        try:
            # Lấy tọa độ vai
            left_shoulder_x = landmarks[LEFT_SHOULDER].x
            right_shoulder_x = landmarks[RIGHT_SHOULDER].x
            
            # Lấy tọa độ mắt
            left_eye_x = landmarks[LEFT_EYE].x
            right_eye_x = landmarks[RIGHT_EYE].x
            nose_x = landmarks[NOSE].x
            
            # Tính trung tâm
            shoulder_center = (left_shoulder_x + right_shoulder_x) / 2
            eye_center = (left_eye_x + right_eye_x) / 2
            
            # Tính offset (độ lệch giữa mặt và vai)
            offset = eye_center - shoulder_center
            
            # Threshold để xác định có đang quay hay không
            TURN_THRESHOLD = 0.04
            
            if abs(offset) > TURN_THRESHOLD:
                # Đang quay
                if offset > 0:
                    return 'right'  # Quay phải
                else:
                    return 'left'   # Quay trái
            else:
                # Chính diện
                return 'front'
        
        except (IndexError, AttributeError):
            # Fallback: dùng độ rộng vai
            try:
                shoulder_diff = abs(right_shoulder_x - left_shoulder_x)
                
                if shoulder_diff > 0.2:  # Vai rộng -> chính diện
                    return 'front'
                else:
                    # Vai nào gần camera hơn (giá trị x nhỏ hơn)
                    if left_shoulder_x < right_shoulder_x:
                        return 'right'
                    else:
                        return 'left'
            except:
                return None
    
    def crop_video_to_square(self, input_video, output_video):
        """
        Crop video về tỷ lệ 1:1 (vuông) từ giữa video
        
        Args:
            input_video: Đường dẫn video input
            output_video: Đường dẫn video output
        
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            cap = cv2.VideoCapture(input_video)
            
            if not cap.isOpened():
                print(f"       ❌ Không thể mở video: {input_video}")
                return False
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Xác định kích thước crop vuông
            side = min(width, height)
            
            # Tính tọa độ crop ở giữa
            x_start = (width - side) // 2
            y_start = (height - side) // 2
            
            print(f"       📐 Kích thước gốc: {width}x{height} -> Crop: {side}x{side}")
            
            # Tạo video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (side, side))
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Crop frame
                cropped = frame[y_start:y_start+side, x_start:x_start+side]
                out.write(cropped)
                
                # Hiển thị tiến độ mỗi 30 frames
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"       ⏳ Đang crop: {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
            cap.release()
            out.release()
            
            print(f"       ✅ Crop hoàn tất: {frame_count} frames")
            return True
            
        except Exception as e:
            print(f"       ❌ Lỗi khi crop video: {e}")
            return False
    
    def process_and_cleanup_folder(self, root_path, sample_frames=30, video_extensions=None, dry_run=False):
        """
        Quét, xử lý và dọn dẹp các video trong cây thư mục:
        1. Phát hiện orientation
        2. XÓA các video không phải FRONT
        3. CROP video FRONT về tỷ lệ 1:1
        4. THAY THẾ video cũ bằng video đã crop
        
        Args:
            root_path: Đường dẫn folder gốc
            sample_frames: Số frames để phân tích mỗi video
            video_extensions: List các đuôi file video
            dry_run: Nếu True, chỉ hiển thị những gì sẽ làm mà không thực sự thực hiện
        
        Returns:
            dict: Kết quả xử lý
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
        
        root_path = Path(root_path)
        
        if not root_path.exists():
            raise ValueError(f"Đường dẫn không tồn tại: {root_path}")
        
        print("="*80)
        print(f"🔍 BẮT ĐẦU XỬ LÝ THƯ MỤC: {root_path}")
        if dry_run:
            print("⚠️  DRY RUN MODE - Chỉ hiển thị, không thực sự xóa/crop")
        print("="*80 + "\n")
        
        # Tìm tất cả các folder con
        subfolders = [f for f in root_path.rglob('*') if f.is_dir()]
        
        if not subfolders:
            print("⚠️  Không tìm thấy folder con nào!")
            return {
                'total_folders': 0,
                'total_videos': 0,
                'deleted_videos': 0,
                'cropped_videos': 0,
                'errors': 0
            }
        
        print(f"📁 Tìm thấy {len(subfolders)} folder con\n")
        
        stats = {
            'total_folders': len(subfolders),
            'total_videos': 0,
            'deleted_videos': 0,
            'cropped_videos': 0,
            'errors': 0,
            'folders_without_front': []
        }
        
        for idx, folder in enumerate(subfolders, 1):
            print(f"\n[{idx}/{len(subfolders)}] 📂 Đang xử lý: {folder.relative_to(root_path)}")
            print("-" * 80)
            
            # Tìm tất cả video trong folder này
            videos = []
            for ext in video_extensions:
                videos.extend(folder.glob(f'*{ext}'))
            
            if not videos:
                print("   ℹ️  Không có video nào trong folder này")
                continue
            
            print(f"   📹 Tìm thấy {len(videos)} video")
            stats['total_videos'] += len(videos)
            
            # Phân tích và xử lý từng video
            has_front = False
            
            for video_idx, video_path in enumerate(videos, 1):
                print(f"\n   [{video_idx}/{len(videos)}] 📹 {video_path.name}")
                
                try:
                    # Detect orientation
                    orientation = self.detect_orientation(
                        str(video_path), 
                        sample_frames=sample_frames,
                        verbose=False
                    )
                    
                    icon_map = {
                        'front': '✅',
                        'left': '◀️',
                        'right': '▶️',
                        'unknown': '❓'
                    }
                    icon = icon_map.get(orientation, '❓')
                    print(f"       {icon} Orientation: {orientation.upper()}")
                    
                    if orientation == 'front':
                        has_front = True
                        
                        # CROP video về 1:1
                        print(f"       🔄 Đang crop video về tỷ lệ 1:1...")
                        
                        # Tạo tên file tạm
                        temp_output = video_path.parent / f"temp_cropped_{video_path.name}"
                        
                        if not dry_run:
                            # Crop video
                            success = self.crop_video_to_square(str(video_path), str(temp_output))
                            
                            if success and temp_output.exists():
                                # Xóa video gốc
                                print(f"       🗑️  Xóa video gốc...")
                                video_path.unlink()
                                
                                # Đổi tên video đã crop thành tên gốc
                                print(f"       ✅ Thay thế bằng video đã crop")
                                temp_output.rename(video_path)
                                
                                stats['cropped_videos'] += 1
                            else:
                                print(f"       ❌ Crop thất bại")
                                stats['errors'] += 1
                                # Xóa file temp nếu tồn tại
                                if temp_output.exists():
                                    temp_output.unlink()
                        else:
                            print(f"       [DRY RUN] Sẽ crop và thay thế video")
                            stats['cropped_videos'] += 1
                    
                    else:
                        # Không phải FRONT -> XÓA
                        print(f"       🗑️  Video không phải FRONT -> Sẽ xóa")
                        
                        if not dry_run:
                            video_path.unlink()
                            print(f"       ✅ Đã xóa")
                        else:
                            print(f"       [DRY RUN] Sẽ xóa video này")
                        
                        stats['deleted_videos'] += 1
                
                except Exception as e:
                    print(f"       ❌ Lỗi: {e}")
                    stats['errors'] += 1
            
            if not has_front:
                stats['folders_without_front'].append(str(folder.relative_to(root_path)))
                print(f"\n   ⚠️  CẢNH BÁO: Folder này KHÔNG CÓ video FRONT nào!")
        
        # Tổng kết
        print("\n" + "="*80)
        print("📊 KẾT QUẢ XỬ LÝ")
        print("="*80)
        print(f"Tổng số folder đã xử lý: {stats['total_folders']}")
        print(f"Tổng số video: {stats['total_videos']}")
        print(f"Video FRONT đã crop: {stats['cropped_videos']}")
        print(f"Video đã xóa (không phải FRONT): {stats['deleted_videos']}")
        print(f"Lỗi: {stats['errors']}")
        print(f"Folder không có FRONT: {len(stats['folders_without_front'])}")
        
        if stats['folders_without_front']:
            print(f"\n⚠️  DANH SÁCH FOLDER KHÔNG CÓ VIDEO FRONT:")
            print("-" * 80)
            for folder_name in stats['folders_without_front']:
                print(f"   ❌ {folder_name}")
        
        print("\n" + "="*80)
        
        return stats
    
    def save_report(self, stats, output_file='processing_report.txt'):
        """
        Lưu báo cáo kết quả ra file text
        
        Args:
            stats: Kết quả từ process_and_cleanup_folder()
            output_file: Tên file output
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BÁO CÁO XỬ LÝ VIDEO\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Tổng số folder đã xử lý: {stats['total_folders']}\n")
            f.write(f"Tổng số video: {stats['total_videos']}\n")
            f.write(f"Video FRONT đã crop: {stats['cropped_videos']}\n")
            f.write(f"Video đã xóa (không phải FRONT): {stats['deleted_videos']}\n")
            f.write(f"Lỗi: {stats['errors']}\n")
            f.write(f"Folder không có FRONT: {len(stats['folders_without_front'])}\n\n")
            
            f.write("="*80 + "\n")
            f.write(f"DANH SÁCH FOLDER KHÔNG CÓ VIDEO FRONT\n")
            f.write("="*80 + "\n")
            
            for folder_name in stats['folders_without_front']:
                f.write(f"❌ {folder_name}\n")
        
        print(f"\n💾 Báo cáo đã được lưu vào: {output_file}")


# ========== SỬ DỤNG ==========

if __name__ == "__main__":
    detector = VideoOrientationDetector()
    
    # Đường dẫn folder gốc chứa các folder con
    root_folder = "/workspace/bosung"  # THAY ĐỔI ĐƯỜNG DẪN NÀY
    
    try:
        print("\n⚠️  CẢNH BÁO: Script này sẽ:")
        print("   1. XÓA tất cả video không phải FRONT (left/right)")
        print("   2. CROP video FRONT về tỷ lệ 1:1")
        print("   3. THAY THẾ video gốc bằng video đã crop")
        print("\n🔍 Chạy DRY RUN trước để xem những gì sẽ xảy ra...\n")
        
        # BƯỚC 1: DRY RUN - Chỉ xem không làm gì
        print("="*80)
        print("BƯỚC 1: DRY RUN (Chỉ xem, không thực hiện)")
        print("="*80 + "\n")
        
        stats_dry = detector.process_and_cleanup_folder(
            root_folder,
            sample_frames=30,
            dry_run=False  # Chỉ xem, không thực sự xóa/crop
        )
        
        # Hỏi xác nhận
        print("\n" + "="*80)
        response = input("\n❓ Bạn có muốn THỰC SỰ XỬ LÝ (xóa và crop) không? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            # BƯỚC 2: THỰC SỰ XỬ LÝ
            print("\n" + "="*80)
            print("BƯỚC 2: THỰC HIỆN XỬ LÝ")
            print("="*80 + "\n")
            
            stats = detector.process_and_cleanup_folder(
                root_folder,
                sample_frames=30,
                dry_run=True  # Thực sự xóa và crop
            )
            
            # Lưu báo cáo
            detector.save_report(stats, output_file='processing_report.txt')
            
            print(f"\n🎯 HOÀN TẤT!")
            print(f"   Video đã xóa: {stats['deleted_videos']}")
            print(f"   Video đã crop: {stats['cropped_videos']}")
            
        else:
            print("\n❌ Đã hủy xử lý. Không có thay đổi nào được thực hiện.")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
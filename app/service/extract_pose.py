import cv2
import numpy as np
from scipy.interpolate import interp1d
import json

# Thử import MediaPipe với error handling
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    print(f"MediaPipe version: {mp.__version__}")
except ImportError:
    MP_AVAILABLE = False
    print("MediaPipe not installed. Please install: pip install mediapipe")

class SignLanguagePoseTransition:
    def __init__(self):
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe is required. Install with: pip install mediapipe")

        # Khởi tạo MediaPipe
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            print("MediaPipe initialized successfully!")
        except AttributeError:
            # Thử cách import khác cho phiên bản cũ
            try:
                from mediapipe.python.solutions import pose as mp_pose
                from mediapipe.python.solutions import hands as mp_hands
                from mediapipe.python.solutions import drawing_utils as mp_drawing
                self.mp_pose = mp_pose
                self.mp_hands = mp_hands
                self.mp_drawing = mp_drawing
                print("MediaPipe initialized (legacy mode)!")
            except:
                raise ImportError(
                    "Cannot initialize MediaPipe. "
                    "Please reinstall: pip uninstall mediapipe && pip install mediapipe"
                )

    def extract_poses_from_video(self, video_path):
        """Trích xuất tất cả các pose từ video"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            return []

        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {total_frames} frames, {fps} fps")

        poses = []
        frame_count = 0

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose, self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}")

                # Convert BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Detect pose
                pose_results = pose.process(image)
                hand_results = hands.process(image)

                # Lưu landmarks
                frame_data = {
                    'pose': None,
                    'hands': [],
                    'frame_number': frame_count
                }

                if pose_results.pose_landmarks:
                    frame_data['pose'] = [
                        [lm.x, lm.y, lm.z, lm.visibility]
                        for lm in pose_results.pose_landmarks.landmark
                    ]

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        frame_data['hands'].append([
                            [lm.x, lm.y, lm.z]
                            for lm in hand_landmarks.landmark
                        ])

                poses.append(frame_data)

        cap.release()
        print(f"Extracted {len(poses)} poses from {video_path}")
        return poses

    def get_skeleton_dimensions(self, pose_landmarks):
        """Tính chiều cao và chiều rộng của khung xương"""
        if not pose_landmarks:
            return None

        # Chuyển sang numpy array để dễ tính toán
        points = np.array([[lm[0], lm[1]] for lm in pose_landmarks])

        # Tính bounding box
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])

        width = max_x - min_x
        height = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        return {
            'width': width,
            'height': height,
            'center': (center_x, center_y),
            'bbox': (min_x, max_x, min_y, max_y)
        }

    def normalize_pose_scale(self, pose_data, target_height, target_width, target_center):
        """Chuẩn hóa kích thước và vị trí của pose"""
        if not pose_data['pose']:
            return pose_data

        # Lấy kích thước hiện tại
        dims = self.get_skeleton_dimensions(pose_data['pose'])
        if not dims:
            return pose_data

        current_width = dims['width']
        current_height = dims['height']
        current_center = dims['center']

        # Tính tỷ lệ scale - dùng chiều cao làm chuẩn để giữ tỷ lệ
        scale = target_height / current_height if current_height > 0 else 1.0

        # Normalize pose landmarks
        normalized_pose = []
        for lm in pose_data['pose']:
            # Dịch về gốc tọa độ (0,0)
            x = (lm[0] - current_center[0]) * scale + target_center[0]
            y = (lm[1] - current_center[1]) * scale + target_center[1]
            z = lm[2] * scale  # Scale cả chiều sâu

            normalized_pose.append([x, y, z, lm[3]])

        # Normalize hands
        normalized_hands = []
        for hand in pose_data['hands']:
            normalized_hand = []
            for lm in hand:
                x = (lm[0] - current_center[0]) * scale + target_center[0]
                y = (lm[1] - current_center[1]) * scale + target_center[1]
                z = lm[2] * scale
                normalized_hand.append([x, y, z])
            normalized_hands.append(normalized_hand)

        return {
            'pose': normalized_pose,
            'hands': normalized_hands,
            'frame_number': pose_data['frame_number']
        }

    def interpolate_poses(self, pose_start, pose_end, num_frames=30):
        """Tạo interpolation giữa 2 poses (đã được normalize trước đó)"""
        interpolated_poses = []

        # Kiểm tra dữ liệu
        if not pose_start['pose'] or not pose_end['pose']:
            print("Warning: Missing pose data for interpolation")
            return []

        # Chuyển poses thành numpy arrays (chỉ lấy x, y, z)
        pose_start_array = np.array([[lm[0], lm[1], lm[2]] for lm in pose_start['pose']])
        pose_end_array = np.array([[lm[0], lm[1], lm[2]] for lm in pose_end['pose']])

        print(f"Creating {num_frames} interpolation frames...")

        # Tạo interpolation cho từng landmark
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0  # 0 to 1

            # Smooth interpolation using cosine (ease in-out)
            smooth_t = (1 - np.cos(t * np.pi)) / 2

            interpolated_pose = pose_start_array * (1 - smooth_t) + pose_end_array * smooth_t

            frame_data = {
                'pose': [[p[0], p[1], p[2], 1.0] for p in interpolated_pose],
                'hands': [],
                'frame_number': i
            }

            # Interpolate hands nếu có
            if pose_start['hands'] and pose_end['hands']:
                for hand_idx in range(min(len(pose_start['hands']), len(pose_end['hands']))):
                    hand_start = np.array(pose_start['hands'][hand_idx])
                    hand_end = np.array(pose_end['hands'][hand_idx])
                    interpolated_hand = hand_start * (1 - smooth_t) + hand_end * smooth_t
                    frame_data['hands'].append(interpolated_hand.tolist())

            interpolated_poses.append(frame_data)

        return interpolated_poses

    def render_pose_video(self, poses, output_path, fps=30, width=720, height=720):
        """Render poses thành video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Rendering {len(poses)} frames to video...")

        for idx, pose_data in enumerate(poses):
            if (idx + 1) % 30 == 0:
                print(f"Rendering frame {idx + 1}/{len(poses)}")

            # Tạo frame đen
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Vẽ pose skeleton
            if pose_data['pose']:
                self._draw_pose_openpose_style(frame, pose_data['pose'], width, height)

            # Vẽ hands
            if pose_data['hands']:
                for hand in pose_data['hands']:
                    self._draw_hand(frame, hand, width, height)

            out.write(frame)

        out.release()
        print(f"Video saved to: {output_path}")

    def _draw_pose_openpose_style(self, frame, landmarks, width, height):
        """Vẽ pose skeleton theo kiểu OpenPose - thân là đường thẳng dọc"""

        # Định nghĩa connections theo kiểu OpenPose
        connections = [
            # Head/Face (đơn giản hóa)
            (0, 1), (1, 2), (2, 3),  # Left eye
            (0, 4), (4, 5), (5, 6),  # Right eye
            (0, 9), (9, 10),         # Mouth

            # Left arm
            (11, 13), (13, 15),  # Vai trái -> khuỷu -> cổ tay
            (15, 17), (15, 19), (15, 21),  # Cổ tay -> ngón tay

            # Right arm
            (12, 14), (14, 16),  # Vai phải -> khuỷu -> cổ tay
            (16, 18), (16, 20), (16, 22),  # Cổ tay -> ngón tay

            # Hông
            (23, 24),  # Hông trái sang hông phải

            # Left leg
            (23, 25), (25, 27),  # Hông -> đầu gối -> mắt cá
            (27, 29), (27, 31),  # Mắt cá -> bàn chân

            # Right leg
            (24, 26), (26, 28),  # Hông -> đầu gối -> mắt cá
            (28, 30), (28, 32),  # Mắt cá -> bàn chân
        ]

        # Màu sắc theo kiểu OpenPose
        colors = {
            'head': (255, 255, 0),      # Cyan cho đầu
            'body': (0, 255, 255),      # Yellow cho thân
            'left_arm': (0, 255, 0),    # Green cho tay trái
            'right_arm': (255, 0, 0),   # Blue cho tay phải
            'left_leg': (0, 200, 255),  # Orange cho chân trái
            'right_leg': (255, 0, 200), # Pink cho chân phải
        }

        def get_point(idx):
            """Lấy tọa độ điểm"""
            if idx < len(landmarks):
                lm = landmarks[idx]
                if len(lm) >= 4 and lm[3] > 0.5:  # Check visibility
                    return (int(lm[0] * width), int(lm[1] * height))
            return None

        def draw_line(p1, p2, color, thickness=2):
            """Vẽ đường nối"""
            if p1 and p2:
                cv2.line(frame, p1, p2, color, thickness)

        # VẼ ĐƯỜNG VAI - RÚT NGẮN 15%
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)

        if left_shoulder and right_shoulder:
            # Tính điểm giữa vai
            mid_x = (left_shoulder[0] + right_shoulder[0]) // 2
            mid_y = (left_shoulder[1] + right_shoulder[1]) // 2
            mid_point = (mid_x, mid_y)

            # Tính vector từ giữa tới mỗi vai
            left_vec_x = left_shoulder[0] - mid_x
            left_vec_y = left_shoulder[1] - mid_y
            right_vec_x = right_shoulder[0] - mid_x
            right_vec_y = right_shoulder[1] - mid_y

            # Rút ngắn 15% = chỉ giữ lại 85% chiều dài
            scale = 0.85

            # Tính vị trí vai mới (ngắn hơn)
            new_left_shoulder = (
                int(mid_x + left_vec_x * scale),
                int(mid_y + left_vec_y * scale)
            )
            new_right_shoulder = (
                int(mid_x + right_vec_x * scale),
                int(mid_y + right_vec_y * scale)
            )

            # Vẽ đường vai ngắn hơn
            cv2.line(frame, new_left_shoulder, new_right_shoulder, colors['body'], 3)

        # Vẽ THÂN - Đường thẳng dọc từ giữa vai xuống giữa hông
        left_hip = get_point(23)
        right_hip = get_point(24)

        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Tính điểm giữa vai
            mid_shoulder = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2
            )
            # Tính điểm giữa hông
            mid_hip = (
                (left_hip[0] + right_hip[0]) // 2,
                (left_hip[1] + right_hip[1]) // 2
            )
            # Vẽ đường thẳng thân
            cv2.line(frame, mid_shoulder, mid_hip, colors['body'], 3)

        # Vẽ các connections khác với màu phù hợp
        for start_idx, end_idx in connections:
            p1 = get_point(start_idx)
            p2 = get_point(end_idx)

            # Xác định màu dựa vào vị trí
            if start_idx in [0, 1, 2, 3, 4, 5, 6, 9, 10]:
                color = colors['head']
            elif start_idx in [11, 13, 15, 17, 19, 21]:
                color = colors['left_arm']
            elif start_idx in [12, 14, 16, 18, 20, 22]:
                color = colors['right_arm']
            elif start_idx in [23, 25, 27, 29, 31]:
                color = colors['left_leg']
            elif start_idx in [24, 26, 28, 30, 32]:
                color = colors['right_leg']
            else:
                color = colors['body']

            draw_line(p1, p2, color, 2)

        # Vẽ các điểm khớp (joints)
        for idx, landmark in enumerate(landmarks):
            if len(landmark) >= 4 and landmark[3] > 0.5:
                x, y = int(landmark[0] * width), int(landmark[1] * height)

                # Điểm lớn hơn cho các khớp chính
                if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]:
                    radius = 5
                else:
                    radius = 3

                cv2.circle(frame, (x, y), radius, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), radius+1, (0, 0, 0), 1)

    def _draw_hand(self, frame, landmarks, width, height):
        """Vẽ hand skeleton"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        ]

        # Vẽ connections
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                start_point = (int(start[0] * width), int(start[1] * height))
                end_point = (int(end[0] * width), int(end[1] * height))

                cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

        # Vẽ joints
        for landmark in landmarks:
            x, y = int(landmark[0] * width), int(landmark[1] * height)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 4, (0, 0, 0), 1)

    def create_multi_video_transition(self, video_paths, output_path, transition_frames=30):
        """
        Tạo video chuyển tiếp từ NHIỀU video

        Args:
            video_paths: List các đường dẫn video (VD: ['video1.mp4', 'video2.mp4', 'video3.mp4'])
            output_path: Đường dẫn file output
            transition_frames: Số frame chuyển tiếp giữa mỗi cặp video
        """
        print("\n" + "="*60)
        print("SIGN LANGUAGE MULTI-VIDEO TRANSITION - OpenPose Style")
        print("="*60 + "\n")

        if len(video_paths) < 2:
            print("ERROR: Cần ít nhất 2 video để tạo transition!")
            return

        print(f"Số video cần xử lý: {len(video_paths)}")
        print(f"Transition frames giữa mỗi video: {transition_frames}\n")

        # BƯỚC 1: Trích xuất poses từ TẤT CẢ video
        all_video_poses = []
        print("="*60)
        print("BƯỚC 1: TRÍCH XUẤT POSES TỪ TẤT CẢ VIDEO")
        print("="*60)

        for idx, video_path in enumerate(video_paths, 1):
            print(f"\n[{idx}/{len(video_paths)}] Extracting poses from: {video_path}")
            poses = self.extract_poses_from_video(video_path)

            if not poses:
                print(f"ERROR: Không thể trích xuất poses từ video {idx}!")
                return

            all_video_poses.append(poses)
            print(f"✓ Video {idx}: {len(poses)} frames extracted")

        # BƯỚC 2: Tính kích thước chuẩn từ TẤT CẢ video
        print("\n" + "="*60)
        print("BƯỚC 2: TÍNH KÍCH THƯỚC CHUẨN (REFERENCE DIMENSIONS)")
        print("="*60)

        all_valid_dims = []
        for video_idx, poses in enumerate(all_video_poses, 1):
            print(f"Analyzing video {video_idx}...")
            for pose in poses:
                dims = self.get_skeleton_dimensions(pose['pose'])
                if dims:
                    all_valid_dims.append(dims)

        if not all_valid_dims:
            print("ERROR: Không thể tính kích thước!")
            return

        # Kích thước và vị trí chuẩn (trung bình của TẤT CẢ video)
        reference_width = sum(d['width'] for d in all_valid_dims) / len(all_valid_dims)
        reference_height = sum(d['height'] for d in all_valid_dims) / len(all_valid_dims)
        reference_center_x = sum(d['center'][0] for d in all_valid_dims) / len(all_valid_dims)
        reference_center_y = sum(d['center'][1] for d in all_valid_dims) / len(all_valid_dims)

        reference_dims = {
            'width': reference_width,
            'height': reference_height,
            'center': (reference_center_x, reference_center_y)
        }

        print(f"\n✓ Reference dimensions calculated:")
        print(f"  Width: {reference_width:.3f}")
        print(f"  Height: {reference_height:.3f}")
        print(f"  Center: ({reference_center_x:.3f}, {reference_center_y:.3f})")

        # BƯỚC 3: Normalize TẤT CẢ video
        print("\n" + "="*60)
        print("BƯỚC 3: NORMALIZE TẤT CẢ VIDEO")
        print("="*60)

        all_normalized_poses = []
        for video_idx, poses in enumerate(all_video_poses, 1):
            print(f"\n[{video_idx}/{len(all_video_poses)}] Normalizing video {video_idx} ({len(poses)} frames)...")

            normalized_poses = []
            for i, pose in enumerate(poses):
                if (i + 1) % 50 == 0:
                    print(f"  Frame {i + 1}/{len(poses)}")

                normalized = self.normalize_pose_scale(
                    pose,
                    reference_dims['height'],
                    reference_dims['width'],
                    reference_dims['center']
                )
                normalized_poses.append(normalized)

            all_normalized_poses.append(normalized_poses)
            print(f"✓ Video {video_idx} normalized: {len(normalized_poses)} frames")

        # BƯỚC 4: Tạo transitions và ghép video
        print("\n" + "="*60)
        print("BƯỚC 4: TẠO TRANSITIONS VÀ GHÉP VIDEO")
        print("="*60)

        final_poses = []

        for i in range(len(all_normalized_poses)):
            # Thêm video hiện tại
            print(f"\n[{i+1}/{len(all_normalized_poses)}] Adding video {i+1} ({len(all_normalized_poses[i])} frames)")
            final_poses.extend(all_normalized_poses[i])

            # Tạo transition nếu không phải video cuối cùng
            if i < len(all_normalized_poses) - 1:
                print(f"Creating transition from video {i+1} to video {i+2}...")

                last_pose_current = all_normalized_poses[i][-1]
                first_pose_next = all_normalized_poses[i + 1][0]

                transition_poses = self.interpolate_poses(
                    last_pose_current,
                    first_pose_next,
                    transition_frames
                )

                if transition_poses:
                    final_poses.extend(transition_poses)
                    print(f"✓ Transition created: {len(transition_poses)} frames")
                else:
                    print(f"⚠ Warning: Could not create transition between video {i+1} and {i+2}")

        # BƯỚC 5: Render video cuối cùng
        print("\n" + "="*60)
        print("BƯỚC 5: RENDER VIDEO CUỐI CÙNG")
        print("="*60)

        total_original_frames = sum(len(poses) for poses in all_normalized_poses)
        total_transition_frames = transition_frames * (len(all_normalized_poses) - 1)

        print(f"\nStatistics:")
        print(f"  Total videos: {len(video_paths)}")
        print(f"  Original frames: {total_original_frames}")
        print(f"  Transition frames: {total_transition_frames}")
        print(f"  Final total frames: {len(final_poses)}")
        print(f"\nRendering to: {output_path}")

        self.render_pose_video(final_poses, output_path)

        print("\n" + "="*60)
        print(f"✓✓✓ SUCCESS! Video saved to: {output_path}")
        print("="*60 + "\n")


# Sử dụng
# if __name__ == "__main__":
#     try:
#         processor = SignLanguagePoseTransition()

#         # ========================================
#         # CẤU HÌNH: THAY ĐỔI ĐƯỜNG DẪN Ở ĐÂY
#         # ========================================

#         # CÁCH 1: Nối 2 video (như cũ)
#         # video_paths = [
#         #    "/content/processed_16b7e27d.mp4",
#         #    "/content/processed_447d90b8.mp4",
#         #    "/content/processed_6046280f.mp4",
#         #    "/content/processed_bbd5319b.mp4",
#         #    "/content/video_3956_fd2392d7.mp4"
#         # ]

#         # CÁCH 2: Nối 3 video
#         # video_paths = [
#         #     "/path/to/video1.mp4",
#         #     "/path/to/video2.mp4",
#         #     "/path/to/video3.mp4"
#         # ]

#         # CÁCH 3: Nối nhiều video (4, 5, 10...)
#         video_paths = [
#             "/content/processed_1bc8cdcf.mp4",
#             "/content/processed_31569beb.mp4",
#             "/content/processed_447d90b8.mp4",
#             "/content/processed_608dfb1b.mp4",
#             "/content/processed_94f5ccc5.mp4",
#             "/content/processed_ca862990.mp4"
#         ]

#         output_path = "multi_transition_output.mp4"

#         # Số frame chuyển tiếp giữa mỗi cặp video (30 frames = 1 giây nếu fps=30)
#         transition_frames = 30

#         # ========================================
#         # CHẠY CHƯƠNG TRÌNH
#         # ========================================

#         processor.create_multi_video_transition(
#             video_paths=video_paths,
#             output_path=output_path,
#             transition_frames=transition_frames
#         )

#     except Exception as e:
#         print(f"\nERROR: {str(e)}")
#         print("\nTroubleshooting:")
#         print("1. Make sure MediaPipe is installed: pip install mediapipe")
#         print("2. Update OpenCV: pip install --upgrade opencv-python")
#         print("3. Check video file paths are correct")
#         print("4. Try reinstalling: pip uninstall mediapipe && pip install mediapipe")
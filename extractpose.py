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

    def detect_body_orientation(self, poses):
        """
        Detect hướng cơ thể (facing left vs right) dựa vào vị trí vai và mắt
        Returns: 'left' nếu quay trái, 'right' nếu quay phải, 'front' nếu chính diện
        """
        orientations = []
        
        for pose_data in poses[:min(30, len(poses))]:  # Lấy 30 frames đầu để phân tích
            if not pose_data['pose']:
                continue
                
            landmarks = pose_data['pose']
            
            # Indices cho MediaPipe Pose
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_EYE = 2
            RIGHT_EYE = 5
            NOSE = 0
            
            # Lấy tọa độ vai
            if LEFT_SHOULDER < len(landmarks) and RIGHT_SHOULDER < len(landmarks):
                left_shoulder_x = landmarks[LEFT_SHOULDER][0]
                right_shoulder_x = landmarks[RIGHT_SHOULDER][0]
                
                # Lấy tọa độ mắt
                left_eye_x = landmarks[LEFT_EYE][0] if LEFT_EYE < len(landmarks) else None
                right_eye_x = landmarks[RIGHT_EYE][0] if RIGHT_EYE < len(landmarks) else None
                nose_x = landmarks[NOSE][0] if NOSE < len(landmarks) else None
                
                # Tính shoulder width và eye distance
                shoulder_diff = abs(right_shoulder_x - left_shoulder_x)
                
                # Phân tích hướng dựa vào vai
                # Nếu vai trái xa hơn vai phải (từ góc nhìn camera) -> quay phải
                # Nếu vai phải xa hơn vai trái -> quay trái
                shoulder_center = (left_shoulder_x + right_shoulder_x) / 2
                
                if left_eye_x and right_eye_x and nose_x:
                    eye_center = (left_eye_x + right_eye_x) / 2
                    
                    # So sánh vị trí mắt và vai để xác định hướng
                    offset = eye_center - shoulder_center
                    
                    # Nếu offset lớn (mặt lệch so với vai) -> đang quay
                    print("offset mắt ", offset)
                    if abs(offset) > 0.04:  # threshold
                        if offset > 0:
                            orientations.append('right')
                        else:
                            orientations.append('left')
                    else:
                        orientations.append('front')
                else:
                    print("shoulder_diff ",shoulder_diff)
                    # Fallback: dựa vào độ rộng vai
                    if shoulder_diff > 0.2:  # Vai rộng -> chính diện
                        orientations.append('front')
                    else:
                        # Dựa vào vai nào gần camera hơn
                        if left_shoulder_x < right_shoulder_x:
                            orientations.append('right')
                        else:
                            orientations.append('left')
        print("orientations",orientations)
        print("===============")
        if not orientations:
            return 'front'
        
        # Lấy orientation phổ biến nhất
        orientation = max(set(orientations), key=orientations.count)
        print(f"  Detected orientation: {orientation} (samples: {len(orientations)})")
        return orientation

    def flip_pose_horizontally(self, pose_data):
        """
        Lật ngang pose (mirror horizontally) - swap left/right landmarks
        """
        if not pose_data['pose']:
            return pose_data
        
        # MediaPipe Pose landmark swap pairs (left <-> right)
        swap_pairs = [
            (2, 5),   # Eyes
            (4, 6),   # Ears  
            (1, 3),   # Eye inner/outer
            (11, 12), # Shoulders
            (13, 14), # Elbows
            (15, 16), # Wrists
            (17, 18), # Pinkies
            (19, 20), # Index fingers
            (21, 22), # Thumbs
            (23, 24), # Hips
            (25, 26), # Knees
            (27, 28), # Ankles
            (29, 30), # Heels
            (31, 32), # Foot indices
        ]
        
        # Clone pose
        flipped_pose = [lm[:] for lm in pose_data['pose']]
        
        # Flip x coordinates (1.0 - x)
        for i in range(len(flipped_pose)):
            flipped_pose[i][0] = 1.0 - flipped_pose[i][0]
        
        # Swap left/right landmarks
        for left_idx, right_idx in swap_pairs:
            if left_idx < len(flipped_pose) and right_idx < len(flipped_pose):
                flipped_pose[left_idx], flipped_pose[right_idx] = \
                    flipped_pose[right_idx][:], flipped_pose[left_idx][:]
        
        # Flip hands
        flipped_hands = []
        for hand in pose_data['hands']:
            flipped_hand = []
            for lm in hand:
                flipped_hand.append([1.0 - lm[0], lm[1], lm[2]])
            flipped_hands.append(flipped_hand)
        
        # Reverse hands order (left hand becomes right hand)
        flipped_hands = flipped_hands[::-1]
        
        return {
            'pose': flipped_pose,
            'hands': flipped_hands,
            'frame_number': pose_data['frame_number']
        }

    def align_video_orientations(self, poses1, poses2):
        """
        Align orientations của 2 videos - flip video 2 nếu cần
        """
        print("\n" + "="*50)
        print("DETECTING AND ALIGNING VIDEO ORIENTATIONS")
        print("="*50)
        
        print("\nAnalyzing Video 1 orientation...")
        orientation1 = self.detect_body_orientation(poses1)
        
        print("\nAnalyzing Video 2 orientation...")
        orientation2 = self.detect_body_orientation(poses2)
        
        print(f"\nVideo 1: {orientation1}")
        print(f"Video 2: {orientation2}")
        
        # Xác định có cần flip không
        need_flip = False
        
        # Case 1: Một video front, một video side -> flip side video
        if orientation1 == 'front' and orientation2 in ['left', 'right']:
            need_flip = True
            print("\n⚠️  Video 1 is front-facing, Video 2 is side view")
            print("   → Flipping Video 2 to match")
        elif orientation1 in ['left', 'right'] and orientation2 == 'front':
            need_flip = True
            print("\n⚠️  Video 1 is side view, Video 2 is front-facing")
            print("   → Flipping Video 2 to match")
        # Case 2: Cả 2 đều side nhưng hướng ngược nhau
        elif (orientation1 == 'left' and orientation2 == 'right') or \
             (orientation1 == 'right' and orientation2 == 'left'):
            need_flip = True
            print("\n⚠️  Videos are facing opposite directions")
            print("   → Flipping Video 2 to match")
        else:
            print("\n✓ Videos have compatible orientations - no flip needed")
        
        # Flip video 2 nếu cần
        if need_flip:
            print("\nFlipping Video 2 horizontally...")
            poses2_flipped = []
            for i, pose in enumerate(poses2):
                if (i + 1) % 50 == 0:
                    print(f"  Flipping frame {i + 1}/{len(poses2)}")
                flipped = self.flip_pose_horizontally(pose)
                poses2_flipped.append(flipped)
            print("✓ Flip complete!")
            return poses1, poses2_flipped
        
        return poses1, poses2

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

    def normalize_pose_scale(self, pose_data, target_height, target_width, target_center=None):
        """Chuẩn hóa kích thước khung xương về target size"""
        if not pose_data['pose']:
            return pose_data

        # Lấy kích thước hiện tại
        dims = self.get_skeleton_dimensions(pose_data['pose'])
        if not dims:
            return pose_data

        current_width = dims['width']
        current_height = dims['height']
        current_center = dims['center']

        # Nếu không chỉ định target_center, giữ nguyên vị trí hiện tại
        if target_center is None:
            target_center = current_center

        # Tính tỷ lệ scale
        scale_x = target_width / current_width if current_width > 0 else 1.0
        scale_y = target_height / current_height if current_height > 0 else 1.0

        # Sử dụng scale_y cho cả chiều ngang và dọc để giữ tỷ lệ aspect ratio
        scale = scale_y

        # Tạo pose mới đã được scale và center
        normalized_pose = []
        for lm in pose_data['pose']:
            # Dịch về gốc tọa độ
            x = lm[0] - current_center[0]
            y = lm[1] - current_center[1]

            # Scale
            x = x * scale
            y = y * scale

            # Dịch về vị trí target center
            x = x + target_center[0]
            y = y + target_center[1]

            normalized_pose.append([x, y, lm[2], lm[3]])

        # Normalize hands tương tự
        normalized_hands = []
        for hand in pose_data['hands']:
            normalized_hand = []
            for lm in hand:
                # Dịch về gốc tọa độ
                x = lm[0] - current_center[0]
                y = lm[1] - current_center[1]

                # Scale
                x = x * scale
                y = y * scale

                # Dịch về vị trí target center
                x = x + target_center[0]
                y = y + target_center[1]

                normalized_hand.append([x, y, lm[2]])
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

    def render_pose_video(self, poses, output_path, fps=30, width=640, height=480):
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
        """Vẽ pose skeleton theo kiểu OpenPose"""

        connections = [
            # Head/Face
            (0, 1), (1, 2), (2, 3),
            (0, 4), (4, 5), (5, 6),
            (0, 9), (9, 10),

            # Neck to shoulders
            (11, 12),

            # Left arm
            (11, 13), (13, 15),
            (15, 17), (15, 19), (15, 21),

            # Right arm
            (12, 14), (14, 16),
            (16, 18), (16, 20), (16, 22),

            # Hông
            (23, 24),

            # Left leg
            (23, 25), (25, 27),
            (27, 29), (27, 31),

            # Right leg
            (24, 26), (26, 28),
            (28, 30), (28, 32),
        ]

        colors = {
            'head': (255, 255, 0),
            'body': (0, 255, 255),
            'left_arm': (0, 255, 0),
            'right_arm': (255, 0, 0),
            'left_leg': (0, 200, 255),
            'right_leg': (255, 0, 200),
        }

        def get_point(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                if len(lm) >= 4 and lm[3] > 0.5:
                    return (int(lm[0] * width), int(lm[1] * height))
            return None

        # Vẽ thân
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)
        left_hip = get_point(23)
        right_hip = get_point(24)

        if left_shoulder and right_shoulder and left_hip and right_hip:
            mid_shoulder = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2
            )
            mid_hip = (
                (left_hip[0] + right_hip[0]) // 2,
                (left_hip[1] + right_hip[1]) // 2
            )
            cv2.line(frame, mid_shoulder, mid_hip, colors['body'], 3)

        # Vẽ connections
        for start_idx, end_idx in connections:
            p1 = get_point(start_idx)
            p2 = get_point(end_idx)

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

            if p1 and p2:
                cv2.line(frame, p1, p2, color, 2)

        # Vẽ joints
        for idx, landmark in enumerate(landmarks):
            if len(landmark) >= 4 and landmark[3] > 0.5:
                x, y = int(landmark[0] * width), int(landmark[1] * height)
                radius = 5 if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26] else 3
                cv2.circle(frame, (x, y), radius, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), radius+1, (0, 0, 0), 1)

    def _draw_hand(self, frame, landmarks, width, height):
        """Vẽ hand skeleton"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                start_point = (int(start[0] * width), int(start[1] * height))
                end_point = (int(end[0] * width), int(end[1] * height))
                cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

        for landmark in landmarks:
            x, y = int(landmark[0] * width), int(landmark[1] * height)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 4, (0, 0, 0), 1)

    def _calculate_average_video_dimensions(self, poses):
        """Tính kích thước trung bình từ tất cả poses trong video"""
        valid_dims = []

        for pose in poses:
            dims = self.get_skeleton_dimensions(pose['pose'])
            if dims:
                valid_dims.append(dims)

        if not valid_dims:
            return None

        avg_width = sum(d['width'] for d in valid_dims) / len(valid_dims)
        avg_height = sum(d['height'] for d in valid_dims) / len(valid_dims)
        avg_center_x = sum(d['center'][0] for d in valid_dims) / len(valid_dims)
        avg_center_y = sum(d['center'][1] for d in valid_dims) / len(valid_dims)

        return {
            'width': avg_width,
            'height': avg_height,
            'center': (avg_center_x, avg_center_y)
        }

    def create_transition_video(self, video1_path, video2_path, output_path,
                                transition_frames=30):
        """Tạo video chuyển tiếp hoàn chỉnh"""
        print("\n" + "="*50)
        print("SIGN LANGUAGE POSE TRANSITION - OpenPose Style")
        print("="*50 + "\n")

        print("Step 1: Extracting poses from video 1 (REFERENCE)...")
        poses1 = self.extract_poses_from_video(video1_path)

        print("\nStep 2: Extracting poses from video 2...")
        poses2 = self.extract_poses_from_video(video2_path)

        if not poses1 or not poses2:
            print("ERROR: Cannot extract poses from videos!")
            return

        # **NEW: Align orientations trước khi normalize**
        poses1, poses2 = self.align_video_orientations(poses1, poses2)

        # Lấy kích thước chuẩn từ video 1
        print("\nStep 3: Calculating reference dimensions from video 1...")
        reference_dims = self._calculate_average_video_dimensions(poses1)
        if not reference_dims:
            print("ERROR: Cannot calculate reference dimensions!")
            return

        print(f"Reference dimensions - Width: {reference_dims['width']:.3f}, Height: {reference_dims['height']:.3f}")

        # Normalize video 2
        print(f"\nStep 4: Normalizing all {len(poses2)} poses from video 2...")
        poses2_normalized = []
        for i, pose in enumerate(poses2):
            if (i + 1) % 50 == 0:
                print(f"Normalizing frame {i + 1}/{len(poses2)}")
            normalized = self.normalize_pose_scale(
                pose,
                reference_dims['height'],
                reference_dims['width'],
                reference_dims['center']
            )
            poses2_normalized.append(normalized)

        # Tạo transition
        last_pose_vid1 = poses1[-1]
        first_pose_vid2 = poses2_normalized[0]

        print(f"\nStep 5: Creating {transition_frames} transition frames...")
        transition_poses = self.interpolate_poses(
            last_pose_vid1,
            first_pose_vid2,
            transition_frames
        )

        if not transition_poses:
            print("ERROR: Cannot create transition!")
            return

        # Kết hợp
        all_poses = poses1 + transition_poses + poses2_normalized

        print(f"\nStep 6: Rendering final video...")
        print(f"Total frames: {len(all_poses)}")
        self.render_pose_video(all_poses, output_path)

        print("\n" + "="*50)
        print(f"SUCCESS! Video saved to: {output_path}")
        print("="*50 + "\n")


# Sử dụng
if __name__ == "__main__":
    try:
        processor = SignLanguagePoseTransition()

        # Đường dẫn video
        video1_path = "/content/202512240836 (1).mp4"
        video2_path = "/content/202512240836.mp4"
        output_path = "transition_output2.mp4"

        # Tạo video chuyển tiếp
        processor.create_transition_video(
            video1_path,
            video2_path,
            output_path,
            transition_frames=30
        )

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure MediaPipe is installed: pip install mediapipe")
        print("2. Update OpenCV: pip install --upgrade opencv-python")
        print("3. Check video file paths are correct")
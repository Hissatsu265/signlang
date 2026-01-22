import cv2
import numpy as np
import json

class OpenPoseVideoRenderer:
    def __init__(self):
        self.colors = {
            'head': (255, 255, 0),
            'body': (0, 255, 255),
            'left_arm': (0, 255, 0),
            'right_arm': (255, 0, 0),
            'left_leg': (0, 200, 255),
            'right_leg': (255, 0, 200),
        }

        # OpenPose keypoint indices
        self.NECK = 1
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 2
        self.MID_HIP = 8
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 8

        # Ngưỡng khoảng cách tối đa (normalized coordinates)
        self.MAX_HAND_BONE_LENGTH = 0.15
        self.MAX_BODY_BONE_LENGTH = 0.3

    def calculate_distance(self, point1, point2):
        """Tính khoảng cách Euclidean giữa 2 điểm"""
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return np.sqrt(dx*dx + dy*dy)

    def is_valid_connection(self, point1, point2, max_distance):
        """
        Kiểm tra xem connection có hợp lệ không dựa trên khoảng cách
        """
        distance = self.calculate_distance(point1, point2)
        return distance < max_distance

    def load_openpose_json(self, json_path):
        """Đọc file JSON OpenPose format"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✓ Đã load {len(data)} frames từ {json_path}")
        return data

    def parse_openpose_frame(self, frame_data):
        """Chuyển đổi OpenPose format sang format của code"""
        if not frame_data.get('people') or len(frame_data['people']) == 0:
            return None

        person = frame_data['people'][0]

        pose_kpts = person.get('pose_keypoints_2d', [])
        pose_landmarks = []

        canvas_width = frame_data.get('canvas_width', 720)
        canvas_height = frame_data.get('canvas_height', 720)

        for i in range(0, len(pose_kpts), 3):
            if i + 2 < len(pose_kpts):
                x = pose_kpts[i]
                y = pose_kpts[i + 1]
                conf = pose_kpts[i + 2]

                pose_landmarks.append([
                    x / canvas_width,
                    y / canvas_height,
                    0.0,
                    conf
                ])

        # Parse hands
        hands = []
        for hand_key in ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            hand_kpts = person.get(hand_key, [])
            if hand_kpts:
                hand_landmarks = []
                for i in range(0, len(hand_kpts), 3):
                    if i + 2 < len(hand_kpts):
                        x = hand_kpts[i]
                        y = hand_kpts[i + 1]
                        conf = hand_kpts[i + 2]

                        hand_landmarks.append([
                            x / canvas_width,
                            y / canvas_height,
                            0.0
                        ])

                if hand_landmarks:
                    hands.append(hand_landmarks)

        return {
            'pose': pose_landmarks,
            'hands': hands,
        }

    def get_root_joint(self, pose_landmarks):
        """Tính root joint (trung điểm 2 vai)"""
        if len(pose_landmarks) < 6:
            return None

        left_shoulder = pose_landmarks[self.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[self.RIGHT_SHOULDER]

        if len(left_shoulder) >= 4 and len(right_shoulder) >= 4:
            if left_shoulder[3] > 0.1 and right_shoulder[3] > 0.1:
                root_x = (left_shoulder[0] + right_shoulder[0]) / 2
                root_y = (left_shoulder[1] + right_shoulder[1]) / 2
                return np.array([root_x, root_y])

        return None

    def get_reference_bone_length(self, pose_landmarks):
        """Tính bone length chuẩn (shoulder width)"""
        if len(pose_landmarks) < 6:
            return None

        left_shoulder = pose_landmarks[self.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[self.RIGHT_SHOULDER]

        if len(left_shoulder) >= 4 and len(right_shoulder) >= 4:
            if left_shoulder[3] > 0.1 and right_shoulder[3] > 0.1:
                dx = left_shoulder[0] - right_shoulder[0]
                dy = left_shoulder[1] - right_shoulder[1]
                shoulder_width = np.sqrt(dx*dx + dy*dy)
                return shoulder_width

        return None

    def calculate_normalization_params(self, reference_frame, target_frame):
        """
        Tính toán tham số chuẩn hóa từ frame đầu tiên
        Returns: (translation_vector, scale_factor)
        """
        ref_root = self.get_root_joint(reference_frame['pose'])
        target_root = self.get_root_joint(target_frame['pose'])

        if ref_root is None or target_root is None:
            print("⚠ Warning: Cannot find root joint, using default params")
            return np.array([0.0, 0.0]), 1.0

        translation = ref_root - target_root

        ref_bone_length = self.get_reference_bone_length(reference_frame['pose'])
        target_bone_length = self.get_reference_bone_length(target_frame['pose'])

        if ref_bone_length is None or target_bone_length is None or target_bone_length < 0.01:
            print("⚠ Warning: Cannot calculate bone length, using scale = 1.0")
            scale_factor = 1.0
        else:
            scale_factor = ref_bone_length / target_bone_length

        return translation, scale_factor

    def normalize_frame(self, frame_data, translation, scale_factor, target_root):
        """
        Áp dụng normalization cho một frame
        """
        if not frame_data or not frame_data.get('pose'):
            return frame_data

        normalized_frame = {
            'pose': [],
            'hands': [],
        }

        # Normalize pose landmarks
        for landmark in frame_data['pose']:
            if len(landmark) >= 4:
                x, y = landmark[0], landmark[1]
                x_scaled = (x - target_root[0]) * scale_factor + target_root[0]
                y_scaled = (y - target_root[1]) * scale_factor + target_root[1]
                x_final = x_scaled + translation[0]
                y_final = y_scaled + translation[1]

                normalized_frame['pose'].append([
                    x_final,
                    y_final,
                    landmark[2],
                    landmark[3]
                ])
            else:
                normalized_frame['pose'].append(landmark)

        # Normalize hands
        for hand in frame_data.get('hands', []):
            normalized_hand = []
            for landmark in hand:
                if len(landmark) >= 3:
                    x, y = landmark[0], landmark[1]
                    x_scaled = (x - target_root[0]) * scale_factor + target_root[0]
                    y_scaled = (y - target_root[1]) * scale_factor + target_root[1]
                    x_final = x_scaled + translation[0]
                    y_final = y_scaled + translation[1]

                    normalized_hand.append([
                        x_final,
                        y_final,
                        landmark[2]
                    ])
                else:
                    normalized_hand.append(landmark)

            normalized_frame['hands'].append(normalized_hand)

        return normalized_frame

    def draw_openpose_skeleton(self, frame, landmarks, width, height):
        """
        Vẽ skeleton OpenPose - KHÔNG VẼ PHẦN ĐẦU/CỔ
        Chỉ vẽ từ vai trở xuống
        """
        # Chỉ vẽ các kết nối từ vai trở xuống (bỏ hết keypoint 0, 1, 14, 15, 16, 17)
        connections = [
            # Bỏ: (0, 1), (0, 14), (14, 16), (0, 15), (15, 17),  # Đầu/Cổ
            (1, 2), (1, 5),  # Từ cổ xuống vai
            (2, 3), (3, 4),  # Tay phải
            (5, 6), (6, 7),  # Tay trái
        ]

        def get_point(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                if len(lm) >= 4 and lm[3] > 0.1:
                    screen_point = (int(lm[0] * width), int(lm[1] * height))
                    norm_point = (lm[0], lm[1])
                    return screen_point, norm_point
            return None, None

        left_shoulder_screen, left_shoulder_norm = get_point(5)
        right_shoulder_screen, right_shoulder_norm = get_point(2)

        if left_shoulder_screen and right_shoulder_screen and left_shoulder_norm and right_shoulder_norm:
            # Kiểm tra khoảng cách trước khi vẽ line từ vai xuống dưới
            if self.is_valid_connection(left_shoulder_norm, right_shoulder_norm, self.MAX_BODY_BONE_LENGTH):
                mid_shoulder_x = (left_shoulder_screen[0] + right_shoulder_screen[0]) // 2
                mid_shoulder_y = (left_shoulder_screen[1] + right_shoulder_screen[1]) // 2
                mid_shoulder = (mid_shoulder_x, mid_shoulder_y)
                bottom_point = (mid_shoulder_x, height)
                cv2.line(frame, mid_shoulder, bottom_point, self.colors['body'], 3)

        for start_idx, end_idx in connections:
            p1_screen, p1_norm = get_point(start_idx)
            p2_screen, p2_norm = get_point(end_idx)

            if start_idx in [5, 6, 7]:
                color = self.colors['left_arm']
            elif start_idx in [2, 3, 4]:
                color = self.colors['right_arm']
            else:
                color = self.colors['body']

            # Chỉ vẽ nếu cả 2 điểm tồn tại VÀ khoảng cách hợp lệ
            if p1_screen and p2_screen and p1_norm and p2_norm:
                if self.is_valid_connection(p1_norm, p2_norm, self.MAX_BODY_BONE_LENGTH):
                    cv2.line(frame, p1_screen, p2_screen, color, 3)

        # Vẽ các điểm keypoint - CHỈ TỪ VAI TRỞ XUỐNG (keypoint 2-7)
        for idx, landmark in enumerate(landmarks):
            # Chỉ vẽ keypoint từ 2 đến 7 (vai và tay)
            if 2 <= idx <= 7 and len(landmark) >= 4 and landmark[3] > 0.1:
                x, y = int(landmark[0] * width), int(landmark[1] * height)
                radius = 6

                cv2.circle(frame, (x, y), radius, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), radius+1, (0, 0, 0), 1)

    def draw_hand(self, frame, landmarks, width, height):
        """
        Vẽ bàn tay với kiểm tra khoảng cách
        """
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Ngón cái
            (0, 5), (5, 6), (6, 7), (7, 8),      # Ngón trỏ
            (0, 9), (9, 10), (10, 11), (11, 12), # Ngón giữa
            (0, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út
            (0, 17), (17, 18), (18, 19), (19, 20), # Ngón út
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                start_screen = (int(start[0] * width), int(start[1] * height))
                end_screen = (int(end[0] * width), int(end[1] * height))

                start_norm = (start[0], start[1])
                end_norm = (end[0], end[1])

                # KIỂM TRA KHOẢNG CÁCH TRƯỚC KHI VẼ
                if self.is_valid_connection(start_norm, end_norm, self.MAX_HAND_BONE_LENGTH):
                    cv2.line(frame, start_screen, end_screen, (0, 255, 255), 2)

        # Vẽ các điểm keypoint
        for landmark in landmarks:
            x, y = int(landmark[0] * width), int(landmark[1] * height)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 4, (0, 0, 0), 1)

    def interpolate_keypoints(self, pose_start, pose_end, num_frames=30):
        """Nội suy giữa 2 poses"""
        if not pose_start or not pose_end:
            print("⚠ Warning: Missing pose data for interpolation")
            return []

        interpolated_frames = []

        start_array = np.array(pose_start)
        end_array = np.array(pose_end)

        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0
            smooth_t = (1 - np.cos(t * np.pi)) / 2
            interpolated = start_array * (1 - smooth_t) + end_array * smooth_t
            interpolated_frames.append(interpolated.tolist())

        return interpolated_frames

    def merge_multi_json_videos(self, json_paths, output_path,
                                 transition_frames=30, fps=30, width=720, height=720):
        """
        Ghép NHIỀU file JSON với SKELETON NORMALIZATION
        """

        print("\n" + "="*60)
        print(f"MERGE {len(json_paths)} JSON VIDEOS - WITH NORMALIZATION")
        print("="*60 + "\n")

        if len(json_paths) < 2:
            print("❌ Cần ít nhất 2 video để merge!")
            return

        # BƯỚC 1: Load và parse tất cả videos
        print("Step 1: Loading all videos...")
        all_raw_videos = []

        for i, json_path in enumerate(json_paths):
            print(f"\n  Video {i+1}/{len(json_paths)}: {json_path}")
            frames_data = self.load_openpose_json(json_path)

            if not frames_data:
                print(f"  ❌ Không thể load video {i+1}!")
                return

            parsed_frames = []
            for frame in frames_data:
                parsed = self.parse_openpose_frame(frame)
                if parsed:
                    parsed_frames.append(parsed)

            all_raw_videos.append(parsed_frames)
            print(f"  ✓ Video {i+1}: {len(parsed_frames)} frames")

        # BƯỚC 2: Tính normalization parameters
        print(f"\n{'='*60}")
        print("Step 2: Calculating normalization parameters...")
        print(f"{'='*60}")

        reference_frame = all_raw_videos[0][0]
        reference_root = self.get_root_joint(reference_frame['pose'])

        if reference_root is None:
            print("❌ Không tìm thấy root joint trong reference frame!")
            return

        print(f"\n  Reference (Video 1 - Frame 1):")
        print(f"    Root position: ({reference_root[0]:.4f}, {reference_root[1]:.4f})")

        ref_bone_length = self.get_reference_bone_length(reference_frame['pose'])
        if ref_bone_length:
            print(f"    Shoulder width: {ref_bone_length:.4f}")

        # Tính normalization params cho từng video
        normalization_params = []

        for i, video_frames in enumerate(all_raw_videos):
            if i == 0:
                normalization_params.append({
                    'translation': np.array([0.0, 0.0]),
                    'scale': 1.0,
                    'root': reference_root
                })
                print(f"\n  Video 1 (Reference): No normalization needed")
            else:
                first_frame = video_frames[0]
                target_root = self.get_root_joint(first_frame['pose'])

                if target_root is None:
                    print(f"\n  ⚠ Video {i+1}: Cannot find root, using default params")
                    normalization_params.append({
                        'translation': np.array([0.0, 0.0]),
                        'scale': 1.0,
                        'root': reference_root
                    })
                    continue

                translation, scale = self.calculate_normalization_params(
                    reference_frame,
                    first_frame
                )

                normalization_params.append({
                    'translation': translation,
                    'scale': scale,
                    'root': target_root
                })

                print(f"\n  Video {i+1}:")
                print(f"    Original root: ({target_root[0]:.4f}, {target_root[1]:.4f})")
                print(f"    Translation: ({translation[0]:.4f}, {translation[1]:.4f})")
                print(f"    Scale factor: {scale:.4f}")

        # BƯỚC 3: Normalize tất cả videos
        print(f"\n{'='*60}")
        print("Step 3: Normalizing all videos...")
        print(f"{'='*60}")

        all_normalized_videos = []

        for i, video_frames in enumerate(all_raw_videos):
            params = normalization_params[i]
            normalized_frames = []

            for frame in video_frames:
                normalized = self.normalize_frame(
                    frame,
                    params['translation'],
                    params['scale'],
                    params['root']
                )
                normalized_frames.append(normalized)

            all_normalized_videos.append(normalized_frames)
            print(f"  ✓ Video {i+1} normalized: {len(normalized_frames)} frames")

        # BƯỚC 4: Tạo transitions và ghép
        print(f"\n{'='*60}")
        print("Step 4: Creating transitions and merging...")
        print(f"{'='*60}")

        final_frames = []

        for i in range(len(all_normalized_videos)):
            final_frames.extend(all_normalized_videos[i])
            print(f"\n  ✓ Added video {i+1}: {len(all_normalized_videos[i])} frames")

            if i < len(all_normalized_videos) - 1:
                print(f"  → Creating transition {i+1} → {i+2}...")

                last_frame = all_normalized_videos[i][-1]
                first_frame = all_normalized_videos[i+1][0]

                # Nội suy pose
                pose_interpolated = self.interpolate_keypoints(
                    last_frame['pose'],
                    first_frame['pose'],
                    transition_frames
                )

                # Nội suy hands
                hands_interpolated = []
                if last_frame.get('hands') and first_frame.get('hands'):
                    num_hands = min(len(last_frame['hands']), len(first_frame['hands']))

                    for hand_idx in range(num_hands):
                        hand_interp = self.interpolate_keypoints(
                            last_frame['hands'][hand_idx],
                            first_frame['hands'][hand_idx],
                            transition_frames
                        )
                        hands_interpolated.append(hand_interp)

                for j in range(transition_frames):
                    transition_frame = {
                        'pose': pose_interpolated[j],
                        'hands': [],
                    }

                    for hand_data in hands_interpolated:
                        if j < len(hand_data):
                            transition_frame['hands'].append(hand_data[j])

                    final_frames.append(transition_frame)

                print(f"     ✓ {transition_frames} transition frames added")

        total_video_frames = sum(len(v) for v in all_normalized_videos)
        total_transition_frames = transition_frames * (len(all_normalized_videos) - 1)

        print(f"\n  Final composition:")
        print(f"    Video frames: {total_video_frames}")
        print(f"    Transition frames: {total_transition_frames}")
        print(f"    Total: {len(final_frames)} frames")

        # BƯỚC 5: Render
        print(f"\n{'='*60}")
        print("Step 5: Rendering final video...")
        print(f"{'='*60}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for idx, parsed_data in enumerate(final_frames):
            if (idx + 1) % 50 == 0:
                print(f"  Rendering frame {idx + 1}/{len(final_frames)}")

            frame = np.zeros((height, width, 3), dtype=np.uint8)

            if parsed_data and parsed_data.get('pose'):
                self.draw_openpose_skeleton(frame, parsed_data['pose'], width, height)

                if parsed_data.get('hands'):
                    for hand in parsed_data['hands']:
                        self.draw_hand(frame, hand, width, height)

            out.write(frame)

        out.release()

        print("\n" + "="*60)
        print(f"✓ MERGE COMPLETED WITH NORMALIZATION!")
        print(f"✓ Total frames: {len(final_frames)}")
        print(f"✓ Video saved: {output_path}")
        print(f"✓ All skeletons normalized (NO FACE/HEAD rendering)")
        print(f"✓ Distance validation applied (hands, body)")
        print("="*60 + "\n")

    def render_video_from_json(self, json_path, output_path, fps=30, width=720, height=720):
        """Tạo video từ file JSON OpenPose"""
        print("\n" + "="*60)
        print("OPENPOSE JSON TO VIDEO RENDERER (NO FACE)")
        print("="*60 + "\n")

        frames_data = self.load_openpose_json(json_path)

        if not frames_data:
            print("❌ Không có dữ liệu!")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Đang render {len(frames_data)} frames...")

        for idx, frame_data in enumerate(frames_data):
            if (idx + 1) % 30 == 0:
                print(f"Rendering frame {idx + 1}/{len(frames_data)}")

            parsed_data = self.parse_openpose_frame(frame_data)
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            if parsed_data and parsed_data['pose']:
                self.draw_openpose_skeleton(frame, parsed_data['pose'], width, height)

                if parsed_data['hands']:
                    for hand in parsed_data['hands']:
                        self.draw_hand(frame, hand, width, height)

            out.write(frame)

        out.release()
        print(f"\n✓ Video saved: {output_path}")


# ============= CÁCH SỬ DỤNG =============

# if __name__ == "__main__":
#     try:
#         renderer = OpenPoseVideoRenderer()

#         # Bạn có thể điều chỉnh các ngưỡng khoảng cách tại đây:
#         renderer.MAX_HAND_BONE_LENGTH = 0.3
#         renderer.MAX_BODY_BONE_LENGTH = 0.5

#         mode = "multi"

#         if mode == "single":
#             json_path = "keypoints.json"
#             renderer.render_video_from_json(
#                 json_path=json_path,
#                 output_path="openpose_skeleton.mp4",
#                 fps=30,
#                 width=720,
#                 height=720
#             )

#         elif mode == "multi":
#             json_paths = [
#                 "/content/nguoithat_00001.json",
#                 # "/content/Wikisign_DGS_process.json",
#                 # "/content/gebaerdenlernende_process.json",
#                 # "/content/gebaerdenlernende_process1.json",
#                 # "/content/haben.json",
#                 # "/content/ich.json",
#                 "/content/PoseKeypoint2_00001fiiiin.json",
#                 "/content/PoseKeypoint2_00001sdfdf.json",
#                 "/content/processed_47c3f3ee.json",
#                 "/content/processed_a336e60c.json",
#                 "/content/processed_ecf447fb.json",
#                 "/content/nguoithat_00001.json",
#             ]
#             renderer.merge_multi_json_videos(
#                 json_paths=json_paths,
#                 output_path="nohead2.mp4",
#                 transition_frames=8,
#                 fps=16,
#                 width=720,
#                 height=720
#             )

#         print("✅ Video đã merge KHÔNG CÓ MẶT/ĐẦU")
#         print("   - Chỉ vẽ từ vai trở xuống")
#         print("   - Distance validation đã được áp dụng")

#     except Exception as e:
#         print(f"\n❌ LỖI: {str(e)}")
#         import traceback
#         traceback.print_exc()
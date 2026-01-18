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
            'face': (255, 200, 255),  # Màu hồng cho mặt
        }

        # OpenPose keypoint indices
        self.NECK = 1
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 2
        self.MID_HIP = 8
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 8

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

        # Parse face keypoints
        face_kpts = person.get('face_keypoints_2d', [])
        face_landmarks = []
        if face_kpts:
            for i in range(0, len(face_kpts), 3):
                if i + 2 < len(face_kpts):
                    x = face_kpts[i]
                    y = face_kpts[i + 1]
                    conf = face_kpts[i + 2]

                    face_landmarks.append([
                        x / canvas_width,
                        y / canvas_height,
                        0.0,
                        conf
                    ])

        return {
            'pose': pose_landmarks,
            'hands': hands,
            'face': face_landmarks
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
            'face': []
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

        # Normalize face
        for landmark in frame_data.get('face', []):
            if len(landmark) >= 4:
                x, y = landmark[0], landmark[1]
                x_scaled = (x - target_root[0]) * scale_factor + target_root[0]
                y_scaled = (y - target_root[1]) * scale_factor + target_root[1]
                x_final = x_scaled + translation[0]
                y_final = y_scaled + translation[1]

                normalized_frame['face'].append([
                    x_final,
                    y_final,
                    landmark[2],
                    landmark[3]
                ])
            else:
                normalized_frame['face'].append(landmark)

        return normalized_frame

    def draw_face(self, frame, landmarks, width, height):
        """
        Vẽ face keypoints theo chuẩn OpenPose (70 điểm)
        """
        if not landmarks or len(landmarks) < 17:
            return

        def get_point(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                if len(lm) >= 4 and lm[3] > 0.1:
                    return (int(lm[0] * width), int(lm[1] * height))
            return None

        # Vẽ đường viền mặt (jawline) - keypoints 0-16
        for i in range(16):
            p1 = get_point(i)
            p2 = get_point(i + 1)
            if p1 and p2:
                cv2.line(frame, p1, p2, self.colors['face'], 1)

        # Vẽ lông mày phải - keypoints 17-21
        for i in range(17, 21):
            p1 = get_point(i)
            p2 = get_point(i + 1)
            if p1 and p2:
                cv2.line(frame, p1, p2, self.colors['face'], 1)

        # Vẽ lông mày trái - keypoints 22-26
        for i in range(22, 26):
            p1 = get_point(i)
            p2 = get_point(i + 1)
            if p1 and p2:
                cv2.line(frame, p1, p2, self.colors['face'], 1)

        # Vẽ sống mũi - keypoints 27-30
        for i in range(27, 30):
            p1 = get_point(i)
            p2 = get_point(i + 1)
            if p1 and p2:
                cv2.line(frame, p1, p2, self.colors['face'], 1)

        # Vẽ mũi dưới - keypoints 31-35
        nose_bottom = [31, 32, 33, 34, 35, 31]  # Tạo vòng khép kín
        for i in range(len(nose_bottom) - 1):
            p1 = get_point(nose_bottom[i])
            p2 = get_point(nose_bottom[i + 1])
            if p1 and p2:
                cv2.line(frame, p1, p2, self.colors['face'], 1)

        # Vẽ mắt phải - keypoints 36-41
        right_eye = [36, 37, 38, 39, 40, 41, 36]  # Vòng khép kín
        for i in range(len(right_eye) - 1):
            p1 = get_point(right_eye[i])
            p2 = get_point(right_eye[i + 1])
            if p1 and p2:
                cv2.line(frame, p1, p2, self.colors['face'], 1)

        # Vẽ mắt trái - keypoints 42-47
        left_eye = [42, 43, 44, 45, 46, 47, 42]  # Vòng khép kín
        for i in range(len(left_eye) - 1):
            p1 = get_point(left_eye[i])
            p2 = get_point(left_eye[i + 1])
            if p1 and p2:
                cv2.line(frame, p1, p2, self.colors['face'], 1)

        # Vẽ môi ngoài - keypoints 48-59
        outer_lip = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48]
        for i in range(len(outer_lip) - 1):
            p1 = get_point(outer_lip[i])
            p2 = get_point(outer_lip[i + 1])
            if p1 and p2:
                cv2.line(frame, p1, p2, self.colors['face'], 1)

        # Vẽ môi trong - keypoints 60-67
        inner_lip = [60, 61, 62, 63, 64, 65, 66, 67, 60]
        for i in range(len(inner_lip) - 1):
            p1 = get_point(inner_lip[i])
            p2 = get_point(inner_lip[i + 1])
            if p1 and p2:
                cv2.line(frame, p1, p2, self.colors['face'], 1)

        # Vẽ các điểm keypoint
        for idx, landmark in enumerate(landmarks):
            if len(landmark) >= 4 and landmark[3] > 0.1:
                x, y = int(landmark[0] * width), int(landmark[1] * height)
                # Vẽ điểm nhỏ hơn một chút
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    def draw_openpose_skeleton(self, frame, landmarks, width, height):
        """Vẽ skeleton OpenPose"""
        connections = [
            (0, 1), (0, 14), (14, 16), (0, 15), (15, 17),
            (1, 2), (1, 5),
            (2, 3), (3, 4),
            (5, 6), (6, 7),
        ]

        def get_point(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                if len(lm) >= 4 and lm[3] > 0.1:
                    return (int(lm[0] * width), int(lm[1] * height))
            return None

        left_shoulder = get_point(5)
        right_shoulder = get_point(2)

        if left_shoulder and right_shoulder:
            mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) // 2
            mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) // 2
            mid_shoulder = (mid_shoulder_x, mid_shoulder_y)
            bottom_point = (mid_shoulder_x, height)
            cv2.line(frame, mid_shoulder, bottom_point, self.colors['body'], 3)

        for start_idx, end_idx in connections:
            p1 = get_point(start_idx)
            p2 = get_point(end_idx)

            if start_idx in [0, 14, 15, 16, 17]:
                color = self.colors['head']
            elif start_idx in [5, 6, 7]:
                color = self.colors['left_arm']
            elif start_idx in [2, 3, 4]:
                color = self.colors['right_arm']
            else:
                color = self.colors['body']

            if p1 and p2:
                cv2.line(frame, p1, p2, color, 3)

        for idx, landmark in enumerate(landmarks):
            if idx <= 7 and len(landmark) >= 4 and landmark[3] > 0.1:
                x, y = int(landmark[0] * width), int(landmark[1] * height)

                if idx in [1, 2, 3, 4, 5, 6, 7]:
                    radius = 6
                else:
                    radius = 4

                cv2.circle(frame, (x, y), radius, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), radius+1, (0, 0, 0), 1)

    def draw_hand(self, frame, landmarks, width, height):
        """Vẽ bàn tay"""
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

                # Nội suy face
                face_interpolated = []
                if last_frame.get('face') and first_frame.get('face'):
                    face_interpolated = self.interpolate_keypoints(
                        last_frame['face'],
                        first_frame['face'],
                        transition_frames
                    )

                for j in range(transition_frames):
                    transition_frame = {
                        'pose': pose_interpolated[j],
                        'hands': [],
                        'face': face_interpolated[j] if j < len(face_interpolated) else []
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

                # VẼ FACE
                if parsed_data.get('face'):
                    self.draw_face(frame, parsed_data['face'], width, height)

            out.write(frame)

        out.release()

        print("\n" + "="*60)
        print(f"✓ MERGE COMPLETED WITH NORMALIZATION!")
        print(f"✓ Total frames: {len(final_frames)}")
        print(f"✓ Video saved: {output_path}")
        print(f"✓ All skeletons normalized to same scale and position")
        print("="*60 + "\n")

    def render_video_from_json(self, json_path, output_path, fps=30, width=720, height=720):
        """Tạo video từ file JSON OpenPose"""
        print("\n" + "="*60)
        print("OPENPOSE JSON TO VIDEO RENDERER")
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

                # VẼ FACE
                if parsed_data.get('face'):
                    self.draw_face(frame, parsed_data['face'], width, height)

            out.write(frame)

        out.release()
        print(f"\n✓ Video saved: {output_path}")


# # ============= CÁCH SỬ DỤNG =============

# if __name__ == "__main__":
#     try:
#         renderer = OpenPoseVideoRenderer()

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
#                 "/content/firstframe1_00001.json",
#                 "/content/processed_6f432c7a.json",
#                 "/content/PoseKeypoint2_00001sdfdf.json",
#                 "/content/PoseKeypoint2_ư00001.json",
#                 "/content/processed_dba7c989.json",
#                 "/content/firstframe1_00001.json",
#             ]
#         renderer.merge_multi_json_videos(
#             json_paths=json_paths,
#             output_path="merged_normalized.mp4",
#             transition_frames=16,
#             fps=16,
#             width=720,
#             height=720
#         )

#         print("✅ Video đã merge với skeleton normalization")
#         print("   Tất cả skeleton đã được chuẩn hóa về cùng tỷ lệ và vị trí")

#     except Exception as e:
#         print(f"\n❌ LỖI: {str(e)}")
#         import traceback
#         traceback.print_exc()
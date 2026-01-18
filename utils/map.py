import os
import re
import random

BASE_PATH = "/workspace/signlang/pose_json"

def get_video_paths_from_text(text):
    # Chuẩn hóa text
    text = text.lower()
    
    # Tách từ trước, giữ lại số
    words = text.split()
    processed_words = []
    
    for word in words:
        # Loại bỏ ký tự đặc biệt nhưng giữ lại số, dấu chấm và dấu trừ
        clean_word = re.sub(r"[^a-zA-Zäöüß0-9.\-]", "", word)
        
        if not clean_word:
            continue
            
        # Kiểm tra xem có phải số không
        is_number = False
        temp_word = clean_word.lstrip('-')  # Bỏ dấu trừ tạm để check
        
        # Check xem có phải số không (bao gồm cả số thập phân)
        if temp_word.replace('.', '', 1).isdigit():
            is_number = True
        
        if is_number:
            # Kiểm tra số âm
            is_negative = clean_word.startswith('-')
            if is_negative:
                processed_words.append('minus')
                clean_word = clean_word[1:]  # Bỏ dấu trừ
            
            # Kiểm tra số thập phân
            if '.' in clean_word:
                parts = clean_word.split('.')
                
                # Phần nguyên
                if parts[0]:  # Nếu có phần nguyên
                    for digit in parts[0]:
                        processed_words.append(digit)
                
                # Dấu chấm -> 'punkt'
                processed_words.append('punkt')
                
                # Phần thập phân
                if len(parts) > 1 and parts[1]:
                    for digit in parts[1]:
                        processed_words.append(digit)
            else:
                # Số nguyên
                try:
                    num = int(clean_word)
                    if num <= 40:
                        # Map trực tiếp
                        processed_words.append(clean_word)
                    else:
                        # Phân rã thành các chữ số
                        for digit in clean_word:
                            processed_words.append(digit)
                except ValueError:
                    processed_words.append(clean_word)
        else:
            # Không phải số, giữ nguyên
            processed_words.append(clean_word)
    
    # Map sang video paths - DÙNG LIST TUPLE
    selected_videos = []
    
    for word in processed_words:
        folder_path = os.path.join(BASE_PATH, word)

        if os.path.isdir(folder_path):
            video_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith("json")
            ]

            if video_files:
                selected_videos.append((word, random.choice(video_files)))
            else:
                selected_videos.append((word, None))
        else:
            selected_videos.append((word, None))

    return selected_videos


# # Ví dụ sử dụng
# if __name__ == "__main__":
#     # Test các trường hợp
#     test_cases = [
#         "Ich habe 25 Euro",           # Số <= 40
#         "Die Zahl ist 123",            # Số > 40 -> 1, 2, 3
#         "Preis 45.99 Euro",            # Số thập phân -> 4, 5, punkt, 9, 9
#         "Temperatur minus 15 Grad",    # Số âm -> minus, 1, 5
#         "Heute -5.5 Grad",             # Số âm thập phân -> minus, 5, punkt, 5
#         "Test 555 hier test",              # Test -> 5, punkt, 1, 5
#         "Preis 5.55 Euro"              # Test -> 5, punkt, 5, 5
#     ]
    
#     for text_input in test_cases:
#         print(f"\nInput: {text_input}")
#         result = get_video_paths_from_text(text_input)
#         for word, path in result:
#             print(f"  {word} -> {path}")
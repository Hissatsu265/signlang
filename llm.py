import transformers
import torch

# Khởi tạo pipeline với model Phi-4
pipeline = transformers.pipeline(
    "text-generation",
    model="microsoft/phi-4",
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto",
)

# Cấu hình System Prompt chuyên biệt cho việc dịch sang tiếng Đức hỗ trợ Sign Language
system_prompt = (
    "You are a linguistic expert specializing in German Sign Language (DGS) glossing. "
    "Your task is to translate English text into simple German sentences. "
    "Rules:\n"
    "1. Use simple, direct German words.\n"
    "2. Maintain a word order that is easy to map to sign language videos (Subject-Object-Verb is preferred).\n"
    "3. Avoid complex grammar, passive voice, or unnecessary articles if they don't add meaning.\n"
    "4. Output ONLY the translated German sentence."
)

# Đoạn text tiếng Anh bạn muốn chuyển đổi
user_input = "I want to eat an apple now."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Translate this to simple German for sign language mapping: {user_input}"},
]

# Chạy mô hình
outputs = pipeline(messages, max_new_tokens=128)

# In kết quả cuối cùng
print("English:", user_input)
print("German (SL optimized):", outputs[0]["generated_text"][-1]['content'])
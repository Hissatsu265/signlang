import transformers
import torch
import re
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time
import gc

class SignLanguageMapper:
    """
    Chuyển đổi câu tiếng Anh/Đức thành chuỗi từ sign language hợp lệ
    """
    
    def __init__(self, video_dir: str, model_name: str = "microsoft/phi-4"):
        """
        Args:
            video_dir: Đường dẫn đến thư mục chứa video sign language
            model_name: Tên model LLM
        """
        self.video_dir = Path(video_dir)
        self.vocabulary = self._load_vocabulary()
        self.model_name = model_name
        self.pipeline = None  # Không load ngay, chỉ load khi cần
        self.fallback_words = list(self.vocabulary)[:50]  # Top 50 từ phổ biến
    
    def __enter__(self):
        """Context manager enter"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - tự động cleanup"""
        self.cleanup()
        return False
    
    def _load_vocabulary(self) -> set:
        """Load danh sách từ có video từ thư mục"""
        vocab = set()
        if self.video_dir.exists():
            vocab = {folder.name.lower() for folder in self.video_dir.iterdir() 
                    if folder.is_dir()}
        return vocab
    
    def _init_llm(self, model_name: str):
        """Khởi tạo LLM pipeline"""
        print("🔄 Loading LLM model into VRAM...")
        return transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )
    
    def _get_pipeline(self):
        """Lazy loading - chỉ load pipeline khi cần dùng"""
        if self.pipeline is None:
            self.pipeline = self._init_llm(self.model_name)
        return self.pipeline
    
    def cleanup(self):
        """Giải phóng VRAM"""
        if self.pipeline is not None:
            print("🧹 Cleaning up VRAM...")
            
            try:
                del self.pipeline
                self.pipeline = None
                
                # Garbage collection
                gc.collect()
                
                # Clear CUDA cache - KHÔNG dùng synchronize
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Bỏ dòng này: torch.cuda.synchronize()
                
                print("✅ VRAM freed successfully")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")
    
    def _get_system_prompt(self) -> str:
        return """
    You are an expert in German Sign Language (DGS) translation.

    Your task:
    Convert an English or German sentence into a SIMPLE German word sequence
    that can be directly mapped to a Sign Language video dataset
    (ONE German word = ONE sign video).

    IMPORTANT CONSTRAINTS:
    - Only output BASE / LEMMA German words (no conjugation, no declension)
    Examples:
    guten, gute, gutes → gut
    habe, hast, hatte → haben
    - Prefer words that are VERY LIKELY to exist in a sign language dictionary
    - Avoid rare, abstract, or grammatical filler words
    - DO NOT invent new words
    - DO NOT explain anything

    LANGUAGE RULES:
    - Follow German Sign Language (DGS) word order
    (Topic / Subject – Object – Verb)
    - Preserve MEANING over grammar
    - Do NOT over-shorten if meaning changes

    CONTEXT DISAMBIGUATION:
    - If a word has multiple meanings (e.g. "have / haben"),
    choose the most semantically correct one based on context
    - If unsure, choose the more COMMON and NEUTRAL sign meaning

    OUTPUT FORMAT:
    - Generate EXACTLY 3 variants
    - Each variant on ONE line
    - Lowercase only
    - Words separated by single spaces
    - No punctuation
    - No numbering

    EXAMPLE:
    Input: "Guten Tag, wie geht es dir?"
    Output:
    gut tag du gehen
    du gut gehen
    du gut

    Input: "I am hungry"
    Output:
    ich hunger
    hunger ich
    hunger

    Generate 3 variants only.
    """
    def _normalize_word(self, word: str) -> str:
        """Chuẩn hóa hình thái đơn giản"""
        rules = {
            "guten": "gut",
            "gute": "gut",
            "gutes": "gut",
            "guter": "gut",
            "geht": "gehen",
            "ging": "gehen",
            "habe": "haben",
            "hast": "haben",
            "hat": "haben",
            "hatte": "haben",
            "bist": "sein",
            "bin": "sein",
            "ist": "sein",
        }
        return rules.get(word, word)
    def _generate_candidates(self, text: str, num_variants: int = 3) -> List[str]:
        """Sinh các câu candidate từ LLM"""
        try:
            pipeline = self._get_pipeline()  # Lazy load
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": text}
            ]
            
            outputs = pipeline(
                messages,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                num_return_sequences=1
            )
            
            response = outputs[0]["generated_text"][-1]["content"]
            candidates = self._parse_candidates(response, num_variants)
            
            # Fallback nếu không parse được
            if not candidates:
                candidates = self._fallback_candidates(text)
                
            return candidates
            
        except Exception as e:
            print(f"LLM error: {e}")
            return self._fallback_candidates(text)
    
    def _parse_candidates(self, response: str, expected: int) -> List[str]:
        
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        candidates = []
        

        for line in lines:
            # Loại bỏ số thứ tự, dấu đầu dòng
            line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
            # Chỉ giữ chữ cái và khoảng trắng
            line = re.sub(r'[^a-zäöüß\s]', '', line.lower())

            if not line:
                continue

            words = line.split()
            words = [self._normalize_word(w) for w in words]
            line = " ".join(words)

            if line:
                candidates.append(line)
        
        return candidates[:expected]
    
    def _fallback_candidates(self, text: str) -> List[str]:
        """Tạo candidates đơn giản khi LLM fail"""
        words = re.findall(r'\b[a-zäöüß]+\b', text.lower())
        return [' '.join(words[:i]) for i in range(len(words), 0, -1)][:5]
    
    def _calculate_coverage(self, sentence: str) -> Tuple[float, List[str]]:
        """
        Tính coverage và tìm từ thiếu
        Returns: (coverage_percentage, missing_words)
        """
        words = sentence.split()
        if not words:
            return 0.0, []
        
        missing = [w for w in words if w not in self.vocabulary]
        coverage = (len(words) - len(missing)) / len(words) * 100
        
        return coverage, missing
    
    def _select_best_candidate(self, candidates: List[str]) -> Tuple[str, List[str]]:
        """Chọn candidate tốt nhất dựa trên coverage"""
        best_sentence = ""
        best_coverage = -1
        best_missing = []
        
        for candidate in candidates:
            coverage, missing = self._calculate_coverage(candidate)
            
            # Ưu tiên: coverage cao nhất, thiếu ít từ nhất
            if coverage > best_coverage:
                best_coverage = coverage
                best_sentence = candidate
                best_missing = missing
        
        return best_sentence, best_missing
    
    def _repair_missing_words(self, sentence: str, missing_words: List[str]) -> str:
        """Sửa các từ thiếu bằng LLM hoặc fallback"""
        words = sentence.split()
        
        for missing_word in missing_words:
            replacement = self._find_replacement(missing_word, sentence)
            
            # Thay thế trong câu
            words = [replacement if w == missing_word else w for w in words]
        
        return ' '.join(words)
    
    def _find_replacement(self, word: str, context: str) -> str:
        """Tìm từ thay thế cho từ thiếu"""
        # Bước 1: Hỏi LLM
        llm_suggestions = self._ask_llm_for_synonym(word, context)
        
        # Bước 2: Check trong vocabulary
        for suggestion in llm_suggestions:
            if suggestion in self.vocabulary:
                return suggestion
        
        # Bước 3: Fallback random từ an toàn
        return random.choice(self.fallback_words)
    
    def _ask_llm_for_synonym(self, word: str, context: str) -> List[str]:
        """Hỏi LLM gợi ý synonym"""
        prompt = f"""Give 4 simple German synonyms for "{word}" in context: "{context}"
Rules:
1. One word per line
2. Only common, simple German words
3. No explanations
4. Format: word1\\nword2\\nword3\\nword4

Synonyms:"""
        
        try:
            pipeline = self._get_pipeline()  # Lazy load
            
            messages = [{"role": "user", "content": prompt}]
            outputs = pipeline(
                messages,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.5
            )
            
            response = outputs[0]["generated_text"][-1]["content"]
            suggestions = self._parse_synonyms(response)
            
            return suggestions if suggestions else self._get_fallback_synonyms(word)
            
        except Exception as e:
            print(f"Synonym generation error: {e}")
            return self._get_fallback_synonyms(word)
    
    def _parse_synonyms(self, response: str) -> List[str]:
        """Parse synonyms từ response"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        synonyms = []
        
        for line in lines:
            # Chỉ lấy từ đầu tiên trong mỗi dòng
            word = re.sub(r'[^a-zäöüß]', '', line.lower())
            if word and len(word) > 2:
                synonyms.append(word)
        
        return synonyms[:4]
    
    def _get_fallback_synonyms(self, word: str) -> List[str]:
        """Fallback synonyms khi LLM fail"""
        # Một số mapping cơ bản
        common_fallbacks = {
            'go': ['gehen', 'laufen'],
            'buy': ['kaufen', 'nehmen'],
            'eat': ['essen'],
            'want': ['wollen', 'möchten'],
            'market': ['markt', 'laden'],
            'later': ['später', 'nachher'],
        }
        
        return common_fallbacks.get(word.lower(), [random.choice(self.fallback_words)])
    
    def process(self, text: str, auto_cleanup: bool = False) -> Dict:
        """
        Hàm chính: Chuyển đổi text thành sign language sentence
        
        Args:
            text: Câu tiếng Anh hoặc tiếng Đức
            auto_cleanup: Tự động giải phóng VRAM sau khi xử lý xong
            
        Returns:
            {
                'original': câu gốc,
                'output': câu sign language hợp lệ,
                'words': list các từ,
                'coverage': % coverage,
                'repaired': có sửa hay không
            }
        """
        try:
            # Bước 1: Generate candidates
            print("text: ", text)
            print("Generate candidates")
            s = time.time()
            candidates = self._generate_candidates(text)
            print("time: ", time.time() - s)
            
            s = time.time()
            print("Select best candidate")
            
            # Bước 2: Select best candidate
            best_sentence, missing_words = self._select_best_candidate(candidates)
            print("time: ", time.time() - s)
            
            print("Repair missing words")
            s = time.time()
            
            # Bước 3: Repair missing words
            repaired = False
            if missing_words:
                best_sentence = self._repair_missing_words(best_sentence, missing_words)
                repaired = True
            print("time: ", time.time() - s)
            
            # Bước 4: Final validation
            final_words = [w for w in best_sentence.split() if w in self.vocabulary]
            final_sentence = ' '.join(final_words) if final_words else random.choice(self.fallback_words)
            
            coverage, _ = self._calculate_coverage(final_sentence)
            
            return {
                'original': text,
                'output': final_sentence,
                'words': final_sentence.split(),
                'coverage': coverage,
                'repaired': repaired
            }
        
        finally:
            # Auto cleanup nếu được yêu cầu
            if auto_cleanup:
                self.cleanup()


# ============ USAGE EXAMPLES ============

def example_with_context_manager():
    """Cách 1: Dùng context manager (KHUYẾN NGHỊ)"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Using Context Manager (Recommended)")
    print("="*60)
    
    test_sentences = [
        "I want to go to the market later to buy apples",
        "How are you today?",
    ]
    
    # Model chỉ load khi bắt đầu with block
    # Tự động cleanup khi thoát khỏi with block
    with SignLanguageMapper(video_dir="/content/drive/MyDrive/1.AIllm/video") as mapper:
        for sentence in test_sentences:
            result = mapper.process(sentence)
            
            print(f"\n📝 Input: {result['original']}")
            print(f"✅ Output: {result['output']}")
            print(f"📊 Coverage: {result['coverage']:.1f}%")
            print(f"🔧 Repaired: {result['repaired']}")
            print("-"*60)
    
    print("✅ VRAM đã được giải phóng tự động")


def example_with_manual_cleanup():
    """Cách 2: Manual cleanup"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Manual Cleanup")
    print("="*60)
    
    mapper = SignLanguageMapper(video_dir="/content/drive/MyDrive/1.AIllm/video")
    
    try:
        result = mapper.process("I want to go to the market")
        print(f"\n📝 Input: {result['original']}")
        print(f"✅ Output: {result['output']}")
    finally:
        mapper.cleanup()  # Giải phóng VRAM thủ công
    
    print("✅ VRAM đã được giải phóng thủ công")


def example_with_auto_cleanup_flag():
    """Cách 3: Dùng auto_cleanup flag"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Using auto_cleanup Flag")
    print("="*60)
    
    mapper = SignLanguageMapper(video_dir="/content/drive/MyDrive/1.AIllm/video")
    
    # auto_cleanup=True sẽ tự động giải phóng sau mỗi lần process
    result = mapper.process("How are you?", auto_cleanup=True)
    
    print(f"\n📝 Input: {result['original']}")
    print(f"✅ Output: {result['output']}")
    print("✅ VRAM đã được giải phóng tự động sau process")


def example_batch_processing():
    """Cách 4: Xử lý nhiều câu rồi mới cleanup"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing")
    print("="*60)
    
    test_sentences = [
        "I want to go to the market later to buy apples",
        "How are you today?",
        "I will have a higher tax in this month",
        "Ich möchte Wasser trinken"
    ]
    
    with SignLanguageMapper(video_dir="/content/drive/MyDrive/1.AIllm/video") as mapper:
        results = []
        
        for sentence in test_sentences:
            result = mapper.process(sentence)  # Không auto_cleanup
            results.append(result)
            
            print(f"\n📝 Input: {result['original']}")
            print(f"✅ Output: {result['output']}")
            print(f"📊 Coverage: {result['coverage']:.1f}%")
            print(f"🎬 Video sequence: {' → '.join([f'{w}.mp4' for w in result['words']])}")
            print("-"*60)
    
    print("✅ Xử lý xong tất cả, VRAM đã được giải phóng")
    return results


def main():
    """Main function - chọn example nào để chạy"""
    
    # Uncomment example bạn muốn test
    
    # example_with_context_manager()  # KHUYẾN NGHỊ
    # example_with_manual_cleanup()
    # example_with_auto_cleanup_flag()
    example_batch_processing()


# if __name__ == "__main__":
#     main()
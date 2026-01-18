import transformers
import torch
import re
import random
import unicodedata
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time
import gc

class SignLanguageMapper:
    """
    Chuyển đổi câu tiếng Anh/Đức thành chuỗi từ sign language hợp lệ
    Chiến lược: Ưu tiên giữ nguyên câu gốc -> Tìm biến thể morphology -> Thay thế từ thiếu -> Tạo câu đồng nghĩa -> Sắp xếp theo DGS
    ** CHUẨN HÓA: Giữ nguyên số trong câu **
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
        self.pipeline = None
        self.fallback_words = list(self.vocabulary)[:50] if self.vocabulary else []

    def __enter__(self):
        """Context manager enter"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - tự động cleanup"""
        self.cleanup()
        return False

    def _extract_numbers(self, text: str) -> Dict[str, str]:
        """
        Trích xuất và thay thế số bằng placeholder
        Returns: dict mapping placeholder -> original number
        """
        number_map = {}
        counter = 0

        def replace_number(match):
            nonlocal counter
            number = match.group()
            placeholder = f"__NUM{counter}__"
            number_map[placeholder] = number
            counter += 1
            return placeholder

        # Pattern để khớp số (bao gồm số thập phân, số âm)
        number_pattern = r'-?\d+(?:[.,]\d+)?'
        modified_text = re.sub(number_pattern, replace_number, text)

        return number_map, modified_text

    def _restore_numbers(self, text: str, number_map: Dict[str, str]) -> str:
        """
        Khôi phục số từ placeholder
        """
        result = text
        for placeholder, original_number in number_map.items():
            result = result.replace(placeholder.lower(), original_number)
            result = result.replace(placeholder, original_number)
        return result

    def _normalize_unicode(self, text: str) -> str:
        """Chuẩn hóa Unicode về dạng NFC (cho tiếng Đức, tiếng Việt)"""
        return unicodedata.normalize('NFC', text).lower().strip()

    def _load_vocabulary(self) -> set:
        """Load danh sách từ có video từ thư mục (với Unicode normalization)"""
        vocab = set()
        if self.video_dir.exists():
            for folder in self.video_dir.iterdir():
                if folder.is_dir():
                    normalized_name = self._normalize_unicode(folder.name)
                    vocab.add(normalized_name)

        # Thêm số vào vocabulary (0-100)
        for i in range(101):
            vocab.add(str(i))

        print(f"📚 Loaded {len(vocab)} words from vocabulary (including numbers)")
        return vocab

    def _word_exists_in_vocab(self, word: str) -> bool:
        """Kiểm tra từ có trong vocabulary (với Unicode normalization)"""
        normalized_word = self._normalize_unicode(word)

        # Kiểm tra nếu là số
        if re.match(r'^-?\d+(?:[.,]\d+)?$', word):
            return True

        return normalized_word in self.vocabulary

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
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("✅ VRAM freed successfully")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")

    def _normalize_word(self, word: str) -> str:
        """Chuyển về base form (lemmatize) - GIỮ NGUYÊN SỐ"""
        # Kiểm tra nếu là số -> giữ nguyên
        if re.match(r'^-?\d+(?:[.,]\d+)?$', word):
            return word

        # Kiểm tra nếu là placeholder số -> giữ nguyên
        if re.match(r'^__NUM\d+__$', word, re.IGNORECASE):
            return word.lower()

        word = self._normalize_unicode(word)

        # German verb conjugations
        rules = {
            # sein
            "bin": "sein", "bist": "sein", "ist": "sein", "sind": "sein", "seid": "sein",
            "war": "sein", "warst": "sein", "waren": "sein", "wart": "sein",
            # haben
            "habe": "haben", "hast": "haben", "hat": "haben", "habt": "haben",
            "hatte": "haben", "hattest": "haben", "hatten": "haben", "hattet": "haben",
            # gehen
            "gehe": "gehen", "gehst": "gehen", "geht": "gehen",
            "ging": "gehen", "gingst": "gehen", "gingen": "gehen", "gingt": "gehen",
            # kommen
            "komme": "kommen", "kommst": "kommen", "kommt": "kommen",
            "kam": "kommen", "kamst": "kommen", "kamen": "kommen", "kamt": "kommen",
            # machen
            "mache": "machen", "machst": "machen", "macht": "machen",
            "machte": "machen", "machtest": "machen", "machten": "machen",
            # sagen
            "sage": "sagen", "sagst": "sagen", "sagt": "sagen",
            "sagte": "sagen", "sagtest": "sagen", "sagten": "sagen",
            # wollen
            "will": "wollen", "willst": "wollen", "wollte": "wollen",
            # können
            "kann": "können", "kannst": "können", "konnte": "können",
            # müssen
            "muss": "müssen", "musst": "müssen", "musste": "müssen",
            # mögen
            "mag": "mögen", "magst": "mögen", "mochte": "mögen",
            # Adjectives
            "guten": "gut", "gute": "gut", "gutes": "gut", "guter": "gut",
            "schönen": "schön", "schöne": "schön", "schönes": "schön",
            "sorgfältig": "sorgfältig", "sorgfältiger": "sorgfältig",
            # English verbs
            "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be",
            "have": "have", "has": "have", "had": "have",
            "do": "do", "does": "do", "did": "do",
            "go": "go", "goes": "go", "went": "go", "gone": "go",
            "come": "come", "comes": "come", "came": "come",
            "want": "want", "wants": "want", "wanted": "want",
            "like": "like", "likes": "like", "liked": "like",
            "love": "love", "loves": "love", "loved": "love",
            "feel": "feel", "feels": "feel", "felt": "feel",
            "feeling": "feel",
        }

        return rules.get(word, word)

    def _get_normalization_prompt(self) -> str:
        """Prompt để chuẩn hóa câu gốc"""
        return """
You are a sign language gloss normalizer.

Your ONLY task: Convert a sentence to sign language format by:

REMOVE:
- Articles (der, die, das, ein, eine, the, a, an)
- Auxiliary verbs (do, does, did, am, is, are, were, have, has, had)
- Punctuation (.,!?;:)

KEEP & CONVERT TO BASE FORM:
- All content words (nouns, verbs, adjectives)
- Question words (wie, was, wer, where, what, who, how, why, when)
- Negation (nicht, kein, no, not)
- Politeness (bitte, please)
- Modal verbs in base form (können, müssen, wollen, can, must, want)
- **NUMBERS AND PLACEHOLDERS (like __NUM0__, __NUM1__) - KEEP EXACTLY AS IS**

PRESERVE:
- Original word order (unless grammatically impossible)
- Sentence type (question stays question, request stays request)
- Core meaning
- **ALL NUMBERS AND NUMBER PLACEHOLDERS (__NUMX__) MUST REMAIN UNCHANGED**

OUTPUT FORMAT:
- Lowercase (except number placeholders which should stay as is)
- Space-separated words
- One line only
- No explanations

EXAMPLES:

Input: "How are you feeling today?"
Output: wie sein du fühlen heute

Input: "I want to go to the market"
Output: ich wollen gehen markt

Input: "Could you please help me?"
Output: können du bitte helfen ich

Input: "I have __NUM0__ apples and __NUM1__ bananas"
Output: ich haben __NUM0__ apfel __NUM1__ banane

Input: "She is __NUM0__ years old"
Output: sie sein __NUM0__ jahr alt

Now normalize this sentence:
"""

    def _normalize_sentence(self, text: str) -> str:
        """Chuẩn hóa câu - GIỮ NGUYÊN SỐ"""
        try:
            # Bước 1: Trích xuất số
            number_map, text_with_placeholders = self._extract_numbers(text)

            if number_map:
                print(f"  🔢 Đã trích xuất {len(number_map)} số: {number_map}")

            # Bước 2: Xử lý với LLM
            pipeline = self._get_pipeline()

            messages = [
                {"role": "system", "content": self._get_normalization_prompt()},
                {"role": "user", "content": text_with_placeholders}
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=150,
                do_sample=False,
                temperature=0.0
            )

            response = outputs[0]["generated_text"][-1]["content"]

            # Parse response - GIỮ NGUYÊN PLACEHOLDER
            # Thay đổi regex để không loại bỏ underscore và chữ hoa trong placeholder
            normalized = re.sub(r'[^a-zäöüß\s_0-9]', '', response.lower().strip())
            words = normalized.split()

            # Apply manual normalization rules (giữ nguyên placeholder)
            words = [self._normalize_word(w) for w in words if w]

            result = ' '.join(words)

            # Bước 3: Khôi phục số
            result = self._restore_numbers(result, number_map)

            return result

        except Exception as e:
            print(f"⚠️ Normalization error: {e}")
            # Fallback cũng phải giữ số
            number_map, text_with_placeholders = self._extract_numbers(text)
            result = self._simple_normalize(text_with_placeholders)
            result = self._restore_numbers(result, number_map)
            return result

    def _simple_normalize(self, text: str) -> str:
        """Fallback normalization nếu LLM fail - GIỮ NGUYÊN SỐ VÀ PLACEHOLDER"""
        # Giữ chữ, số, khoảng trắng, underscore
        text = re.sub(r'[^\w\s]', '', text.lower())

        remove_words = {'der', 'die', 'das', 'ein', 'eine', 'the', 'a', 'an',
                       'do', 'does', 'did', 'am', 'is', 'are', 'was', 'were',
                       'have', 'has', 'had'}

        words = [self._normalize_word(w) for w in text.split()
                if w not in remove_words]

        return ' '.join(words)

    def _calculate_coverage(self, sentence: str) -> Tuple[float, List[str]]:
        """
        Tính coverage và tìm từ thiếu (với Unicode normalization)
        Returns: (coverage_percentage, missing_words)
        """
        words = sentence.split()
        if not words:
            return 0.0, []

        missing = [w for w in words if not self._word_exists_in_vocab(w)]
        coverage = (len(words) - len(missing)) / len(words) * 100

        return coverage, missing

    def _get_morphology_variants_prompt(self) -> str:
        """Prompt để tìm biến thể morphology"""
        return """
You are a linguistic expert in German Morphology and Sign Language (DGS) Lexicon.

CONTEXT:
I have a DGS dataset, but some words from my input sentence are missing. I need you to find a variant of the missing word that is more likely to exist in a standard DGS dictionary.

CRITICAL RULES:
1. **NEVER MODIFY NUMBERS OR NUMBER PLACEHOLDERS** - If the word is a number or __NUMX__, return it as is
2. MANDATORY: Find the "Lemma" (Base form/Infinitiv) of the word. (e.g., "kocht" -> "kochen", "porsche" -> "porsche").
3. ADJECTIVES/ADVERBS: Remove comparative/superlative suffixes (e.g., "schneller" -> "schnell").
4. NOUNS: Convert plural to singular (e.g., "kinder" -> "kind").
5. VERBS: Always return the Infinitive form (e.g., "ging" -> "gehen").
6. NO SYNONYMS: Do not replace the word with a different meaning (e.g., do not change "salty" to "spicy").
7. COMPOUND WORDS: If a compound word is missing, try breaking it down (e.g., "kochunterricht" -> "kochen" + "unterricht").
8. FINGERSPELLING: If no grammatical variant makes sense, return the original word.

INPUT FORMAT:
Missing Word: [word]
Full Sentence Context: [sentence]

OUTPUT FORMAT:
Return a JSON list of potential variants in order of priority (most likely to be in a DGS dataset first).

Example:
Input:
- Missing Word: kocht
- Sentence: Meine Mutter kocht heute Abend.

Output: ["kochen", "kocht"]

Example with number:
Input:
- Missing Word: __NUM0__
- Sentence: Ich habe __NUM0__ Äpfel

Output: ["__NUM0__"]

Now find variants for:
"""

    def _find_morphology_variants(self, word: str, context: str) -> List[str]:
        """
        Tìm các biến thể morphology của từ (lemma, singular, infinitive, etc.)
        GIỮ NGUYÊN SỐ VÀ PLACEHOLDER
        """
        # Nếu là số hoặc placeholder -> giữ nguyên
        if re.match(r'^-?\d+(?:[.,]\d+)?$', word) or re.match(r'^__NUM\d+__$', word, re.IGNORECASE):
            return [word]

        try:
            pipeline = self._get_pipeline()

            prompt = f"""
Missing Word: {word}
Full Sentence Context: {context}
"""

            messages = [
                {"role": "system", "content": self._get_morphology_variants_prompt()},
                {"role": "user", "content": prompt}
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0
            )

            response = outputs[0]["generated_text"][-1]["content"]

            # Parse JSON response
            try:
                # Extract JSON from response
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    variants = json.loads(json_match.group())
                    # Normalize all variants (giữ nguyên số/placeholder)
                    variants = [self._normalize_unicode(v) if not re.match(r'^-?\d+(?:[.,]\d+)?$|^__NUM\d+__$', v, re.IGNORECASE)
                               else v for v in variants if v]
                    return variants
                else:
                    return [word]
            except:
                return [word]

        except Exception as e:
            print(f"    ⚠️ Morphology variant error: {e}")
            return [word]

    def _find_replacement(self, word: str, context: str) -> Optional[str]:
        """
        Tìm từ thay thế cho từ thiếu
        Bước 1: Tìm biến thể morphology
        Bước 2: Nếu không tìm thấy -> tìm từ đồng nghĩa
        GIỮ NGUYÊN SỐ
        """
        # Nếu là số -> không cần thay thế
        if re.match(r'^-?\d+(?:[.,]\d+)?$', word):
            return word

        # BƯỚC 2: Tìm từ đồng nghĩa
        print(f"    🔍 Bước 2: Tìm từ đồng nghĩa...")
        llm_synonyms = self._ask_llm_for_synonym(word, context)

        for synonym in llm_synonyms:
            if self._word_exists_in_vocab(synonym):
                print(f"    ✓ Thay thế bằng từ đồng nghĩa: '{word}' → '{synonym}'")
                return synonym

        print(f"    ✗ Không tìm được từ thay thế cho '{word}'")
        return None

    def _ask_llm_for_synonym(self, word: str, context: str) -> List[str]:
        """
        Hỏi LLM gợi ý từ đồng nghĩa/thay thế
        KHÔNG BAO GIỜ THAY THẾ SỐ
        """
        # Nếu là số -> không tìm synonym
        if re.match(r'^-?\d+(?:[.,]\d+)?$', word):
            return [word]

        prompt = f"""
You are assisting a German Sign Language (DGS) gloss system.

Task: Find SAFE replacement words for a missing word.

Requirements:
- Preserve CORE MEANING
- Use SIMPLE, COMMON words
- Do NOT change sentence type (question/statement)
- Return EXACTLY 4 alternatives
- If NO safe replacement exists, return "NONE" four times
- **NEVER replace numbers or number placeholders**

Missing word: "{word}"
Sentence context: "{context}"

OUTPUT FORMAT (4 lines, one word per line, lowercase, no explanations):
""".strip()

        try:
            pipeline = self._get_pipeline()

            messages = [{"role": "user", "content": prompt}]
            outputs = pipeline(
                messages,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.0
            )

            response = outputs[0]["generated_text"][-1]["content"]
            suggestions = self._parse_synonyms(response)

            # Remove NONE and duplicates, normalize Unicode
            suggestions = [self._normalize_unicode(s) for s in suggestions
                          if s.lower() != "none" and s.lower() != word.lower()]

            return suggestions

        except Exception as e:
            print(f"    ⚠️ Synonym error: {e}")
            return []

    def _parse_synonyms(self, response: str) -> List[str]:
        """Parse synonyms từ response"""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        synonyms = []

        for line in lines:
            word = re.sub(r'[^a-zäöüß]', '', line.lower())
            if word and len(word) > 1:
                synonyms.append(word)

        return synonyms[:4]

    def _try_repair_sentence(self, sentence: str, missing_words: List[str]) -> Tuple[str, bool]:
        """
        Bước 2: Thử thay thế từng từ thiếu (morphology variants + synonyms)
        Returns: (repaired_sentence, fully_repaired)
        GIỮ NGUYÊN SỐ
        """
        print(f"\n  🔧 Thử thay thế {len(missing_words)} từ thiếu...")

        words = sentence.split()
        replacement_count = 0

        for missing_word in missing_words:
            # Bỏ qua số
            if re.match(r'^-?\d+(?:[.,]\d+)?$', missing_word):
                continue

            replacement = self._find_replacement(missing_word, sentence)

            if replacement:
                words = [replacement if w == missing_word else w for w in words]
                replacement_count += 1

        repaired = ' '.join(words)

        # Check if fully repaired
        _, still_missing = self._calculate_coverage(repaired)
        fully_repaired = len(still_missing) == 0

        print(f"  {'✅' if fully_repaired else '⚠️'} Đã thay thế {replacement_count}/{len(missing_words)} từ")

        return repaired, fully_repaired

    def _get_synonym_sentence_prompt(self) -> str:
        """Prompt để tạo câu đồng nghĩa"""
        return """
You are an expert in German Sign Language (DGS) gloss generation.

Task: Generate synonym sentences that:
- Keep EXACTLY THE SAME MEANING as the original
- Use ONLY base form words (lemma)
- Preserve sentence type (question → question, request → request)
- Use SIMPLE, COMMON words
- **PRESERVE ALL NUMBERS AND PLACEHOLDERS (__NUMX__) EXACTLY AS THEY APPEAR**

REMOVE:
- Articles, auxiliaries, punctuation

KEEP:
- Question markers (wie, was, wer, how, what, who)
- Negation (nicht, kein, no, not)
- Politeness (bitte, please)
- Modal meaning (können, müssen, wollen, can, must, want)
- **ALL NUMBERS AND PLACEHOLDERS - DO NOT CHANGE THEM**

OUTPUT: Generate EXACTLY 2 alternative sentences.

EXAMPLES:

Input: "How are you feeling today?"
Output:
wie sein du fühlen heute
wie du heute fühlen

Input: "Could you please help me?"
Output:
können du bitte helfen ich
du helfen ich bitte

Input: "I have __NUM0__ apples"
Output:
ich haben __NUM0__ apfel
ich __NUM0__ apfel haben

Now generate 2 alternatives for:
"""

    def _generate_synonym_sentences(self, normalized_sentence: str, original_text: str) -> List[str]:
        """
        Bước 3: Tạo 2 câu đồng nghĩa nếu câu gốc không thể sửa
        GIỮ NGUYÊN SỐ
        """
        try:
            # Trích xuất số từ câu gốc
            number_map, text_with_placeholders = self._extract_numbers(original_text)

            pipeline = self._get_pipeline()

            messages = [
                {"role": "system", "content": self._get_synonym_sentence_prompt()},
                {"role": "user", "content": text_with_placeholders}
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.4,
                top_p=0.9
            )

            response = outputs[0]["generated_text"][-1]["content"]

            # Parse alternatives
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            alternatives = []

            for line in lines:
                # Clean
                line = re.sub(r'^[\d\-\*\.]+\s*', '', line)
                # Giữ số, chữ, underscore
                line = re.sub(r'[^a-zäöüß\s_0-9]', '', line.lower())

                if not line:
                    continue

                words = [self._normalize_word(w) for w in line.split()]
                if words:
                    alt_sentence = ' '.join(words)
                    # Khôi phục số
                    alt_sentence = self._restore_numbers(alt_sentence, number_map)
                    alternatives.append(alt_sentence)

            return alternatives[:2]

        except Exception as e:
            print(f"⚠️ Synonym generation error: {e}")
            return []

    def _get_dgs_word_order_prompt(self) -> str:
        return """
You are a DGS word-order transformer.

TASK: Reorder ONLY the given words to a common DGS word order.

CRITICAL RULES:
- Use exactly the same words (no adding, removing, changing)
- **DO NOT CHANGE, REMOVE, OR MODIFY ANY NUMBERS**
- Do not change word form or spelling
- No punctuation
- Output one single line only

DGS ORDER GUIDELINES (apply ONLY if elements are present):
1. TIME / CONDITION →
   (heute, morgen, gestern, jetzt, später, oft, immer, nie, ...
    any time, frequency, or conditional words)
2. PLACE →
   (zu-hause, schule, arbeit, berlin, hier, dort, ...
    any location or place-related words)
3. TOPIC or OBJECT (INCLUDING NUMBERS) →
   (brot, apfel, auto, arbeit, problem, 5, 10, 20 ...
    any noun, object, or number that is being discussed)
4. SUBJECT →
   (ich, du, er, sie, wir, ihr, sie, ...
    any person or animate subject)
5. MAIN VERB →
   (essen, gehen, arbeiten, lernen, ...
    any main action verb)
6. MODAL →
   (müssen, wollen, können, sollen, dürfen, ...)
7. NEGATION →
   (nicht, kein, nie, ...)
8. WH-WORD (always last) →
   (was, wer, wo, wann, warum, wie, ...)

**NUMBERS stay in their logical position (usually with the object they quantify)**

If multiple orders are possible, choose the most common DGS order.
Do not explain.

EXAMPLES:
Input: heute ich arbeit müssen
Output: heute arbeit ich müssen

Input: morgen du brot essen nicht
Output: morgen brot du essen nicht

Input: ich haben 5 apfel
Output: ich 5 apfel haben

Input: sie sein 25 jahr alt
Output: sie 25 jahr alt sein

Now reorder:"""

    def _reorder_to_dgs(self, sentence: str) -> str:
        """
        Bước 4: Sắp xếp lại thứ tự từ theo cấu trúc DGS
        GIỮ NGUYÊN SỐ
        """
        try:
            # Trích xuất số
            number_map, text_with_placeholders = self._extract_numbers(sentence)

            pipeline = self._get_pipeline()

            messages = [
                {"role": "system", "content": self._get_dgs_word_order_prompt()},
                {"role": "user", "content": text_with_placeholders}
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.0
            )

            response = outputs[0]["generated_text"][-1]["content"]

            # Clean response - giữ số và underscore
            reordered = re.sub(r'[^a-zäöüß\s_0-9]', '', response.lower().strip())
            reordered = ' '.join(reordered.split())  # Remove extra spaces

            # Khôi phục số
            reordered = self._restore_numbers(reordered, number_map)

            # Validate: check if all words are preserved
            input_words = sorted(sentence.split())
            output_words = sorted(reordered.split())

            if input_words == output_words:
                return reordered
            else:
                print(f"    ⚠️ DGS reordering changed words, keeping original order")
                return sentence

        except Exception as e:
            print(f"⚠️ DGS reordering error: {e}")
            return sentence

    def process(self, text: str, auto_cleanup: bool = False) -> Dict:

        try:
            print(f"\n{'='*60}")
            print(f"📝 Input: {text}")
            print(f"{'='*60}")

            # BƯỚC 1: Chuẩn hóa câu gốc
            print("\n🔄 BƯỚC 1: Chuẩn hóa câu gốc...")
            normalized = self._normalize_sentence(text)
            print(f"  → {normalized}")

            coverage, missing = self._calculate_coverage(normalized)
            print(f"  📊 Coverage: {coverage:.1f}% | Thiếu: {len(missing)} từ")

            # Nếu không thiếu từ → chuyển sang bước 4
            if not missing:
                print("  ✅ Hoàn hảo! Không cần xử lý thêm")
                final_sentence = normalized
                strategy = "normalized"
            else:
                # BƯỚC 2: Thay thế từng từ thiếu (morphology + synonyms)
                print(f"\n🔧 BƯỚC 2: Thay thế từ thiếu...")
                print(f"  Từ thiếu: {missing}")
                repaired, fully_repaired = self._try_repair_sentence(normalized, missing)

                if fully_repaired:
                    coverage, _ = self._calculate_coverage(repaired)
                    print(f"  ✅ Đã sửa xong! Coverage: {coverage:.1f}%")
                    final_sentence = repaired
                    strategy = "repaired"
                else:
                    # BƯỚC 3: Tạo câu đồng nghĩa
                    print(f"\n🔄 BƯỚC 3: Tạo câu đồng nghĩa...")
                    alternatives = self._generate_synonym_sentences(normalized, text)

                    # Thêm câu đã repair vào danh sách candidates
                    all_candidates = [repaired] + alternatives

                    print(f"\n📋 Đang xét {len(all_candidates)} ứng viên:")
                    for i, candidate in enumerate(all_candidates, 1):
                        cov, miss = self._calculate_coverage(candidate)
                        print(f"  {i}. [{cov:.0f}%] {candidate}")
                        if miss:
                            print(f"     Thiếu: {miss}")

                    # Chọn câu có coverage cao nhất
                    best_candidate = max(all_candidates,
                                       key=lambda s: self._calculate_coverage(s)[0])

                    # Lọc bỏ từ không có trong vocabulary (NHƯNG GIỮ SỐ)
                    final_words = [w for w in best_candidate.split() if self._word_exists_in_vocab(w)]

                    if not final_words:
                        print("  ⚠️ Không còn từ nào hợp lệ, dùng fallback")
                        final_words = [random.choice(self.fallback_words)] if self.fallback_words else ["hallo"]

                    final_sentence = ' '.join(final_words)
                    strategy = "synonym"

            # BƯỚC 4: Sắp xếp lại thứ tự từ theo DGS
            print(f"\n🔄 BƯỚC 4: Sắp xếp theo thứ tự DGS...")
            print(f"  Trước khi sắp xếp: {final_sentence}")

            dgs_ordered = self._reorder_to_dgs(final_sentence)
            print(f"  Sau khi sắp xếp: {dgs_ordered}")

            # Tính coverage cuối cùng
            coverage, _ = self._calculate_coverage(dgs_ordered)

            return self._create_result(text, dgs_ordered, coverage, strategy)

        finally:
            if auto_cleanup:
                self.cleanup()

    def _create_result(self, original: str, output: str, coverage: float, strategy: str) -> Dict:
        """Tạo result dict"""
        print(f"\n{'='*60}")
        print(f"🎬 RESULT")
        print(f"{'='*60}")
        print(f"Original : {original}")
        print(f"Output   : {output}")
        print(f"Coverage : {coverage:.1f}%")
        print(f"Strategy : {strategy}")
        print(f"{'='*60}\n")

        return {
            'original': original,
            'output': output,
            'words': output.split(),
            'coverage': coverage,
            'strategy': strategy
        }


def example_batch_processing():
    """Xử lý nhiều câu - Bao gồm test với số"""
    print("\n" + "="*60)
    print("BATCH PROCESSING - Ưu tiên giữ câu gốc + Morphology Variants + GIỮ SỐ")
    print("="*60)

    test_sentences = [
        "Ich werde um 7 Uhr am Pier auf dich warten.",  # Test với số giờ
        "Ich esse einen Apfel",
        "Heute gehe ich zur Arbeit",
        "Ich habe 5 Äpfel und 3 Bananen",  # Test với số
        "Sie ist 25 Jahre alt",  # Test với số
        "Der Lehrer erklärt den Studenten wie das System funktioniert",
        "Wir treffen uns um 15 Uhr",  # Test với số giờ
        "Das Buch kostet 19.99 Euro",  # Test với số thập phân
    ]

    with SignLanguageMapper(video_dir="/workspace/signlang/pose_json") as mapper:
        results = []

        for sentence in test_sentences:
            print("\n" + "="*60)
            print(f"Processing: {sentence}")
            s = time.time()
            result = mapper.process(sentence)
            results.append(result)
            print(f"Time: {time.time()-s:.2f}s")
            print("-"*60)
            time.sleep(0.5)  # Tránh overload

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['strategy']:12} | {r['coverage']:5.1f}% | {r['output']}")

    print("\n✅ Hoàn thành! VRAM đã được giải phóng")
    return results


if __name__ == "__main__":
    example_batch_processing()
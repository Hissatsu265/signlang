import re

def replace_be_haben_with_was(original_sentence, sentence):
    if "was" in original_sentence.lower() and "be" in sentence.lower() :
        sentence = re.sub(r"\bbe\b", "was", sentence, flags=re.IGNORECASE)
    if "geht" in original_sentence.lower() and "gehen" in sentence.lower() and "geht" not in sentence.lower():
        sentence = re.sub(r"\bgehen\b", "geht", sentence, flags=re.IGNORECASE)

    return sentence
def dgs_postprocess(sentence: str,original_sentence:str="") -> str:
    
    sentence = replace_be_haben_with_was(original_sentence, sentence)
    tokens = sentence.lower().split()
    
    # 1. remove always
    REMOVE_ALWAYS = {"haben", "werden", "tun", "lassen", 
    "bleiben","das", "der", "die",
    "ein", "eine",
    "ins", "in","bist", "ist", "habe",
    "dem", "den", "des", "dieser", "diese", "dieses",
    "jeder", "jede", "jedes", "manche", "solcher",
    "zu", "zur", "zum", "von", "vom",
    "über","für","bei","auf","an","es","man",
    "nach", "aus", "mit",
    "be","um", "aber", "und", "oder", "weil", "da", "dass", "denn"
    }

    # 2. modal mapping
    MAP_MODAL = {
        "können": None,
        "sollen": "empfehlung",
        "dürfen": "erlaubt",
        "möchten": "wollen",
        "mögen": "wollen",
    }

    # 3. words that suggest 'sein' is meaningful
    SEIN_IDENTITY_HINT = {
        "arzt", "lehrer", "student", "dozent", "problem",
        "wichtig", "richtig", "falsch", "zufrieden"
    }
    # 4.QUESTION_WORDS
    is_short = len(tokens) <= 8
    is_question = original_sentence.strip().endswith("?")
    QUESTION_WORDS = {
        "was",      # what
        "wer",      # who
        "wo",       # where
        "wann",     # when
        "warum",    # why
        "wie",      # how
        "wohin",    # where to
        "woher",    # where from
        "wie-viel", # how much
        "wieviel",
        "welche",   # which
    }
    
    output = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # remove duplicates (consecutive)
        if output and tok == output[-1]:
            i += 1
            continue

        # remove auxiliaries
        if tok in REMOVE_ALWAYS:
            i += 1
            continue

        # modal handling
        if tok in MAP_MODAL:
            mapped = MAP_MODAL[tok]
            if mapped is not None:
                output.append(mapped)
            i += 1
            continue

        # special handling for 'sein'
        if tok == "sein":
            # keep only if next word indicates identity / emphasis
            if i + 1 < len(tokens) and tokens[i + 1] in SEIN_IDENTITY_HINT:
                output.append("sein")
            i += 1
            continue
        # default: keep token
        output.append(tok)
        i += 1
    # ==============================================
    if is_short or is_question:
        for i, tok in enumerate(output):
            if tok in QUESTION_WORDS:
                # remove and append to end
                output.pop(i)
                output.append(tok)
                break
    # ===============================================
    return " ".join(output)

# inp = "warum haben du du gestern nicht antworten"
# print(dgs_postprocess(inp,inp))

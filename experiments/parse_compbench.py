"""
Robust parser for T2I-CompBench prompts.

Extracts (object, attribute) pairs from compositional text prompts
for BLIP-VQA evaluation of attribute binding (color, shape, texture).

Handles both simple patterns ("a red car and a blue bus") and complex
sentences from the official T2I-CompBench dataset.
"""

import re
from typing import List, Dict, Tuple

# ============================================================
# Attribute vocabularies per category
# ============================================================

COLOR_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "brown", "gray",
    "grey", "orange", "pink", "purple", "gold", "silver", "beige", "tan",
    "stainless", "whitish", "golden", "crimson", "scarlet", "navy",
    "teal", "maroon", "ivory", "cream", "khaki", "aqua", "coral",
    "cyan", "indigo", "lavender", "magenta", "olive", "salmon", "violet",
}

SHAPE_WORDS = {
    "round", "square", "triangular", "rectangular", "circular", "oval",
    "spherical", "cylindrical", "conical", "cubic", "pyramidal",
    "pentagonal", "hexagonal", "oblong", "teardrop", "diamond",
    "crescent", "flat", "tall", "short", "long", "wide", "narrow",
    "thin", "thick", "big", "small", "slim", "deep", "curved",
    "straight", "spiral", "pointed", "arched", "angular", "dome",
}

TEXTURE_WORDS = {
    "fluffy", "smooth", "rough", "wooden", "metal", "metallic", "plastic",
    "leather", "glass", "ceramic", "rubber", "fabric", "cotton", "silk",
    "velvet", "knitted", "woven", "woolen", "furry", "glossy", "matte",
    "stone", "sandy", "rocky", "marble", "bronze", "crystal", "fuzzy",
    "polished", "shiny", "paper", "bamboo", "concrete", "clay",
    "porcelain", "iron", "steel", "chrome", "copper", "tin", "brass",
    "granite", "sandstone", "terracotta", "linen", "nylon", "denim",
    "corduroy", "satin", "chiffon", "suede", "vinyl", "canvas",
    "fleece", "burlap", "wicker", "lacquered", "braided", "mosaic",
    "plaster", "felt", "straw", "tweed", "plush", "ribbed", "frosted",
    "hammered", "pebbled", "patent",
}

ATTRIBUTE_VOCABS = {
    "color": COLOR_WORDS,
    "shape": SHAPE_WORDS,
    "texture": TEXTURE_WORDS,
}


def parse_compbench_prompt(
    prompt: str,
    attribute_type: str,
) -> List[Dict[str, str]]:
    """
    Parse a T2I-CompBench prompt into (object, attribute) pairs.

    Strategy:
      1. Try spaCy-based dependency parsing for robust extraction
      2. Fall back to regex for simple "a ADJ NOUN and a ADJ NOUN" patterns
      3. Fall back to vocabulary-based heuristic matching

    Args:
        prompt: Raw text prompt from T2I-CompBench
        attribute_type: "color", "shape", or "texture"

    Returns:
        List of dicts with "object" and "attribute" keys.
        Empty list only if no pairs can be found at all.
    """
    pairs = _parse_with_spacy(prompt, attribute_type)

    if not pairs:
        pairs = _parse_with_regex(prompt, attribute_type)

    if not pairs:
        pairs = _parse_with_vocab_heuristic(prompt, attribute_type)

    return pairs


def _parse_with_spacy(prompt: str, attribute_type: str) -> List[Dict[str, str]]:
    """Use spaCy dependency parsing to extract adj-noun pairs."""
    try:
        import spacy
    except ImportError:
        return []

    try:
        try:
            import en_core_web_trf
            nlp = en_core_web_trf.load()
        except ImportError:
            nlp = spacy.load("en_core_web_sm")
    except Exception:
        return []

    vocab = ATTRIBUTE_VOCABS.get(attribute_type, set())
    doc = nlp(prompt.lower())
    pairs = []
    seen = set()

    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue
        if token.dep_ in ("amod", "compound") and token.head.pos_ in ("NOUN", "PROPN"):
            continue

        for child in token.children:
            if child.dep_ in ("amod", "nmod", "compound", "npadvmod", "advmod", "acomp"):
                attr_text = child.text.strip()
                if attr_text in vocab:
                    key = (token.text, attr_text)
                    if key not in seen:
                        seen.add(key)
                        pairs.append({"object": token.text, "attribute": attr_text})

    if not pairs:
        for token in doc:
            if token.pos_ == "ADJ" and token.text in vocab:
                head = token.head
                if head.pos_ in ("NOUN", "PROPN"):
                    key = (head.text, token.text)
                    if key not in seen:
                        seen.add(key)
                        pairs.append({"object": head.text, "attribute": token.text})

    return pairs


_ATTR_NOUN_PATTERN = re.compile(
    r'(?:a|an|the)\s+(\w+)\s+((?:\w+\s+)*\w+)',
    re.IGNORECASE,
)


def _parse_with_regex(prompt: str, attribute_type: str) -> List[Dict[str, str]]:
    """Regex-based extraction for common T2I-CompBench patterns."""
    vocab = ATTRIBUTE_VOCABS.get(attribute_type, set())
    pairs = []
    seen = set()

    parts = re.split(r'\s+and\s+|\s*,\s+', prompt, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip().rstrip(".")
        match = _ATTR_NOUN_PATTERN.search(part)
        if match:
            attr_candidate = match.group(1).lower()
            rest = match.group(2).strip().lower()
            rest_words = rest.split()

            if attr_candidate in vocab and rest_words:
                obj = rest_words[-1].strip(".,;:!?()")
                if len(rest_words) > 1:
                    obj = " ".join(rest_words[-2:]).strip(".,;:!?()")
                key = (obj, attr_candidate)
                if key not in seen:
                    seen.add(key)
                    pairs.append({"object": obj, "attribute": attr_candidate})

    return pairs


_ONE_OTHER_PATTERN = re.compile(
    r'(?:one|first)\s+(\w+)\s+and\s+(?:the\s+)?(?:other|second)\s+(\w+)',
    re.IGNORECASE,
)


def _parse_with_vocab_heuristic(prompt: str, attribute_type: str) -> List[Dict[str, str]]:
    """Last-resort: scan for vocabulary words adjacent to nouns."""
    vocab = ATTRIBUTE_VOCABS.get(attribute_type, set())
    pairs = []
    seen = set()

    m = _ONE_OTHER_PATTERN.search(prompt.lower())
    if m:
        w1, w2 = m.group(1).strip(), m.group(2).strip()
        subject_match = re.match(r'^.*?(\w+)\s*,', prompt.lower())
        subject = subject_match.group(1) if subject_match else "object"
        if w1 in vocab:
            pairs.append({"object": subject, "attribute": w1})
        if w2 in vocab:
            pairs.append({"object": subject, "attribute": w2})
        if pairs:
            return pairs

    words = prompt.lower().split()
    for i, word in enumerate(words):
        clean_word = word.strip(".,;:!?()\"'")
        if clean_word in vocab and i + 1 < len(words):
            next_word = words[i + 1].strip(".,;:!?()\"'")
            if next_word not in {"and", "or", "the", "a", "an", "with", "on", "in", "of"}:
                key = (next_word, clean_word)
                if key not in seen:
                    seen.add(key)
                    pairs.append({"object": next_word, "attribute": clean_word})

    return pairs


# ============================================================
# Batch parsing utility
# ============================================================

_spacy_nlp_cache = {}


def parse_prompts_batch(
    prompts: List[str],
    attribute_type: str,
) -> List[Tuple[str, List[Dict[str, str]]]]:
    """
    Parse a batch of prompts, returning (prompt, attr_pairs) tuples.

    Uses spaCy with batched processing for efficiency on large datasets.
    """
    results = []
    for prompt in prompts:
        attrs = parse_compbench_prompt(prompt, attribute_type)
        results.append((prompt, attrs))
    return results


# ============================================================
# Test / debug
# ============================================================

def test_parser():
    """Quick test on representative T2I-CompBench prompts."""
    test_cases = {
        "color": [
            "a green bench and a blue bowl",
            "a red car and a white sheep",
            "A bathroom with white tile and a beige toilet.",
            "a black cat and a brown mouse",
            "a yellow bus and a green tree",
        ],
        "shape": [
            "an oblong cucumber and a teardrop plum",
            "a cubic block and a cylindrical bottle",
            "a tall skyscraper and a short lamppost",
            "a round pizza and a square pizza box",
            "a big elephant and a small dog",
        ],
        "texture": [
            "a plastic toy and a glass bottle",
            "a metallic desk lamp and a fluffy sweater",
            "a rubber ball and a leather wallet",
            "a fabric towel and a glass table",
            "a wooden door and a glass mirror",
        ],
    }

    for attr_type, prompts in test_cases.items():
        print(f"\n=== {attr_type.upper()} ===")
        for prompt in prompts:
            pairs = parse_compbench_prompt(prompt, attr_type)
            pairs_str = ", ".join(
                f"({p['object']}, {p['attribute']})" for p in pairs
            )
            status = "OK" if pairs else "FAIL"
            print(f"  [{status}] \"{prompt}\"")
            print(f"         -> {pairs_str}")


if __name__ == "__main__":
    test_parser()

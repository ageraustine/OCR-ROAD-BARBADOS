"""
Analyze training texts to inform prompt engineering for HTR task.

Extracts patterns, characteristics, and common features from historical documents
to create an optimized prompt that aligns with the actual data distribution.
"""

import pandas as pd
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from data_utils import REPO_ROOT


def analyze_text_patterns(df: pd.DataFrame) -> dict:
    """Analyze patterns in transcription texts."""

    texts = df["Target"].dropna().astype(str)

    analysis = {
        "total_samples": len(texts),
        "statistics": {},
        "patterns": {},
        "vocabulary": {},
        "examples": {},
    }

    # ===== BASIC STATISTICS =====
    lengths = texts.str.len()
    words_per_text = texts.str.split().str.len()

    analysis["statistics"] = {
        "char_length": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
        },
        "word_count": {
            "min": int(words_per_text.min()),
            "max": int(words_per_text.max()),
            "mean": float(words_per_text.mean()),
            "median": float(words_per_text.median()),
        },
    }

    # ===== DOCUMENT TYPE PATTERNS =====
    # Common openings
    openings = Counter()
    for text in texts:
        # First 5 words
        words = text.split()[:5]
        if len(words) >= 3:
            opening = " ".join(words[:3])
            openings[opening] += 1

    analysis["patterns"]["common_openings"] = openings.most_common(10)

    # Common legal phrases
    legal_phrases = [
        r"By this (?:publique|public) (?:Act|act)",
        r"To all Christian [Pp]eople",
        r"Signed[,\s]+[Ss]ealed[,\s]+and delivered",
        r"(?:in the )?p?rsence of",
        r"this present (?:writeing|writing)",
        r"Instrument of protest",
        r"have and (?:to )?h?ould",
        r"grace of God",
        r"one thousand (?:six|seven|eight) hundred",
        r"Charles the (?:Second|First)",
    ]

    phrase_counts = defaultdict(int)
    for phrase_pattern in legal_phrases:
        count = sum(1 for text in texts if re.search(phrase_pattern, text, re.IGNORECASE))
        if count > 0:
            phrase_counts[phrase_pattern] = count

    analysis["patterns"]["legal_phrases"] = dict(phrase_counts)

    # ===== SPELLING VARIATIONS =====
    # Common historical spellings
    variations = {
        "publique/public": (
            sum(1 for text in texts if "publique" in text.lower()),
            sum(1 for text in texts if re.search(r'\bpublic\b', text, re.IGNORECASE))
        ),
        "prsence/presence": (
            sum(1 for text in texts if "prsence" in text.lower()),
            sum(1 for text in texts if "presence" in text.lower())
        ),
        "whome/whom": (
            sum(1 for text in texts if "whome" in text.lower()),
            sum(1 for text in texts if re.search(r'\bwhom\b', text, re.IGNORECASE))
        ),
        "writeing/writing": (
            sum(1 for text in texts if "writeing" in text.lower()),
            sum(1 for text in texts if "writing" in text.lower())
        ),
    }

    analysis["vocabulary"]["spelling_variations"] = variations

    # ===== CAPITALIZATION PATTERNS =====
    # Mid-sentence capitals (legal emphasis)
    mid_caps = Counter()
    for text in texts:
        words = text.split()
        for i, word in enumerate(words[1:], 1):  # Skip first word
            if word and word[0].isupper() and len(word) > 2:
                # Not at sentence start (no period before)
                if i > 0 and not words[i-1].endswith('.'):
                    mid_caps[word] += 1

    analysis["patterns"]["mid_sentence_capitals"] = mid_caps.most_common(20)

    # ===== NUMBERS AND DATES =====
    # Written-out numbers
    number_words = r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b'
    texts_with_numbers = sum(1 for text in texts if re.search(number_words, text, re.IGNORECASE))

    # Years (1600s-1700s pattern)
    year_pattern = r'(?:one|two) thousand (?:six|seven) hundred'
    texts_with_years = sum(1 for text in texts if re.search(year_pattern, text, re.IGNORECASE))

    analysis["patterns"]["numbers"] = {
        "texts_with_written_numbers": texts_with_numbers,
        "texts_with_written_years": texts_with_years,
        "percentage_with_numbers": f"{100 * texts_with_numbers / len(texts):.1f}%",
    }

    # ===== PUNCTUATION PATTERNS =====
    punct_stats = {
        "comma": sum(text.count(',') for text in texts) / len(texts),
        "period": sum(text.count('.') for text in texts) / len(texts),
        "caret (^)": sum(text.count('^') for text in texts),  # Interlineations
        "hyphen": sum(text.count('-') for text in texts) / len(texts),
    }

    analysis["patterns"]["punctuation"] = punct_stats

    # ===== SPECIAL CHARACTERS =====
    # Superscripts, interlineations
    interlineations = sum(1 for text in texts if '^' in text)

    analysis["patterns"]["special_marks"] = {
        "interlineations (^)": interlineations,
        "percentage": f"{100 * interlineations / len(texts):.1f}%",
    }

    # ===== EXAMPLE TEXTS =====
    # Sample short, medium, long
    by_length = texts.sort_values(key=lambda x: x.str.len())

    analysis["examples"]["short"] = by_length.iloc[len(by_length)//10]
    analysis["examples"]["medium"] = by_length.iloc[len(by_length)//2]
    analysis["examples"]["long"] = by_length.iloc[int(len(by_length)*0.95)]

    return analysis


def print_analysis(analysis: dict):
    """Pretty-print analysis results."""

    print("="*80)
    print("TEXT PATTERN ANALYSIS FOR PROMPT ENGINEERING")
    print("="*80)

    print(f"\nTotal samples: {analysis['total_samples']}")

    # Statistics
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)
    stats = analysis["statistics"]
    print(f"\nCharacter length:")
    print(f"  Min: {stats['char_length']['min']}")
    print(f"  Max: {stats['char_length']['max']}")
    print(f"  Mean: {stats['char_length']['mean']:.1f}")
    print(f"  Median: {stats['char_length']['median']:.1f}")

    print(f"\nWord count:")
    print(f"  Min: {stats['word_count']['min']}")
    print(f"  Max: {stats['word_count']['max']}")
    print(f"  Mean: {stats['word_count']['mean']:.1f}")
    print(f"  Median: {stats['word_count']['median']:.1f}")

    # Document patterns
    print("\n" + "="*80)
    print("COMMON DOCUMENT OPENINGS")
    print("="*80)
    for opening, count in analysis["patterns"]["common_openings"]:
        print(f"  {count:4d}x  \"{opening}...\"")

    print("\n" + "="*80)
    print("LEGAL PHRASES (regex matches)")
    print("="*80)
    for phrase, count in analysis["patterns"]["legal_phrases"].items():
        print(f"  {count:4d}x  {phrase}")

    # Spelling variations
    print("\n" + "="*80)
    print("SPELLING VARIATIONS (historical vs modern)")
    print("="*80)
    for var_pair, (hist_count, mod_count) in analysis["vocabulary"]["spelling_variations"].items():
        hist, mod = var_pair.split('/')
        print(f"  {hist:20s} {hist_count:4d}x  |  {mod:20s} {mod_count:4d}x")

    # Capitalization
    print("\n" + "="*80)
    print("MID-SENTENCE CAPITALS (legal emphasis, top 20)")
    print("="*80)
    for word, count in analysis["patterns"]["mid_sentence_capitals"]:
        print(f"  {count:4d}x  {word}")

    # Numbers
    print("\n" + "="*80)
    print("NUMBERS AND DATES")
    print("="*80)
    for key, val in analysis["patterns"]["numbers"].items():
        print(f"  {key:35s}: {val}")

    # Punctuation
    print("\n" + "="*80)
    print("PUNCTUATION (average per text)")
    print("="*80)
    for punct, avg in analysis["patterns"]["punctuation"].items():
        print(f"  {punct:20s}: {avg:.2f}")

    # Special marks
    print("\n" + "="*80)
    print("SPECIAL EDITORIAL MARKS")
    print("="*80)
    for mark, val in analysis["patterns"]["special_marks"].items():
        print(f"  {mark:35s}: {val}")

    # Examples
    print("\n" + "="*80)
    print("EXAMPLE TEXTS (by length)")
    print("="*80)
    print(f"\nShort ({len(analysis['examples']['short'])} chars):")
    print(f"  \"{analysis['examples']['short']}\"")
    print(f"\nMedium ({len(analysis['examples']['medium'])} chars):")
    print(f"  \"{analysis['examples']['medium']}\"")
    print(f"\nLong ({len(analysis['examples']['long'])} chars):")
    print(f"  \"{analysis['examples']['long'][:200]}...\"")


def recommend_prompt(analysis: dict) -> str:
    """Generate recommended prompt based on analysis."""

    print("\n" + "="*80)
    print("RECOMMENDED PROMPT")
    print("="*80)

    # Determine document era from year patterns
    if analysis["patterns"]["numbers"]["texts_with_written_years"] > 10:
        era = "17th-18th century"
    else:
        era = "historical"

    # Check if legal documents dominate
    legal_count = sum(analysis["patterns"]["legal_phrases"].values())
    is_legal = legal_count > len(analysis["patterns"]["legal_phrases"]) * 50

    prompt = f"""Transcribe this {era} handwritten document image exactly as written.

Important guidelines:
- Preserve original spelling (e.g., "publique", "prsence", "whome")
- Maintain capitalization, including mid-sentence capitals
- Keep all punctuation marks exactly as shown
- Preserve special marks like ^ (interlineations)
- Write numbers and dates as they appear
- Do not modernize or correct spelling
- Output only the transcription text"""

    if is_legal:
        prompt += "\n- This is a legal document - accuracy is critical"

    return prompt.strip()


def main():
    # Load training data
    train_csv = REPO_ROOT / "dataset/Train.csv"
    print(f"Loading data from {train_csv}...")
    df = pd.read_csv(train_csv)

    # Analyze patterns
    print("Analyzing text patterns...\n")
    analysis = analyze_text_patterns(df)

    # Print results
    print_analysis(analysis)

    # Recommend prompt
    prompt = recommend_prompt(analysis)
    print(f"\n{prompt}\n")

    # Save recommended prompt
    prompt_file = REPO_ROOT / "src/qwen2vl/recommended_prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    print(f"\n✓ Saved recommended prompt to: {prompt_file}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

"""
Interactive Error Inspector

Load the validation error analysis results and interactively explore specific errors.
Useful for debugging and understanding specific failure modes.

Usage:
    python inspect_errors.py --results validation_error_analysis.csv
"""

import argparse
import re
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def levenshtein(a: str, b: str) -> int:
    """Edit distance."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


def highlight_differences(pred: str, ref: str, context: int = 20) -> str:
    """
    Highlight character-level differences between prediction and reference.

    Returns a formatted string showing differences.
    """
    output = []
    output.append("=" * 80)
    output.append("CHARACTER-LEVEL COMPARISON:")
    output.append("=" * 80)

    # Find first difference
    min_len = min(len(pred), len(ref))
    first_diff = min_len

    for i in range(min_len):
        if pred[i] != ref[i]:
            first_diff = i
            break

    if first_diff == min_len and len(pred) != len(ref):
        # Difference is only at the end (length mismatch)
        output.append(f"\nFirst difference at position {min_len} (length mismatch)")
        output.append(f"Reference length: {len(ref)}")
        output.append(f"Prediction length: {len(pred)}")

        start = max(0, min_len - context)
        end = max(len(pred), len(ref))

        output.append(f"\nContext (position {start}-{end}):")
        output.append(f"REF:  ...{ref[start:end]}")
        output.append(f"PRED: ...{pred[start:end]}")
    elif first_diff < min_len:
        output.append(f"\nFirst difference at position {first_diff}")

        # Show context around difference
        start = max(0, first_diff - context)
        end = min(min_len, first_diff + context)

        output.append(f"\nContext (position {start}-{end}):")
        output.append(f"REF:  ...{ref[start:end]}...")
        output.append(f"PRED: ...{pred[start:end]}...")

        # Highlight the specific difference
        output.append(f"\nDifference at position {first_diff}:")
        output.append(f"  REF:  '{ref[first_diff]}' (char code: {ord(ref[first_diff])})")
        if first_diff < len(pred):
            output.append(f"  PRED: '{pred[first_diff]}' (char code: {ord(pred[first_diff])})")
        else:
            output.append(f"  PRED: [missing]")
    else:
        output.append("\nNo character differences (exact match)")

    return "\n".join(output)


def show_word_differences(pred: str, ref: str) -> str:
    """
    Show word-level differences with alignment.
    """
    pred_words = pred.split()
    ref_words = ref.split()

    output = []
    output.append("\n" + "=" * 80)
    output.append("WORD-LEVEL COMPARISON:")
    output.append("=" * 80)

    # Simple alignment by position
    max_len = max(len(pred_words), len(ref_words))
    differences = []

    for i in range(max_len):
        ref_word = ref_words[i] if i < len(ref_words) else "[missing]"
        pred_word = pred_words[i] if i < len(pred_words) else "[missing]"

        if ref_word != pred_word:
            edit_dist = levenshtein(pred_word, ref_word) if ref_word != "[missing]" and pred_word != "[missing]" else 999

            differences.append({
                "position": i,
                "ref": ref_word,
                "pred": pred_word,
                "edit_distance": edit_dist,
            })

    if differences:
        output.append(f"\n{len(differences)} word differences found:")
        for diff in differences[:10]:  # Show first 10
            output.append(f"\n  Position {diff['position']}:")
            output.append(f"    REF:  '{diff['ref']}'")
            output.append(f"    PRED: '{diff['pred']}'")
            if diff['edit_distance'] < 999:
                output.append(f"    Edit distance: {diff['edit_distance']}")

        if len(differences) > 10:
            output.append(f"\n  ... and {len(differences) - 10} more")
    else:
        output.append("\nNo word differences (perfect match)")

    return "\n".join(output)


def inspect_sample(row: pd.Series, image_dir: Path = None, show_image: bool = False):
    """
    Detailed inspection of a single sample.
    """
    print("\n" + "=" * 80)
    print(f"SAMPLE ID: {row['id']}")
    print("=" * 80)

    # Metadata
    print("\nMETADATA:")
    if "condition_tier" in row and pd.notna(row["condition_tier"]):
        print(f"  Condition: {row['condition_tier']}")
    if "text_type" in row and pd.notna(row["text_type"]):
        print(f"  Text type: {row['text_type']}")
    if "length_tier" in row and pd.notna(row["length_tier"]):
        print(f"  Length tier: {row['length_tier']}")
    if "text_length" in row and pd.notna(row["text_length"]):
        print(f"  Text length: {row['text_length']} chars")

    # Metrics
    print("\nMETRICS:")
    print(f"  CER: {row['cer']:.4f} ({row['char_edits']}/{row['total_chars']} char edits)")
    print(f"  WER: {row['wer']:.4f} ({row['word_edits']}/{row['total_words']} word edits)")
    print(f"  Competition Score (0.5*CER + 0.5*WER): {row['competition_score']:.4f}")

    if "normalized_acc_1" in row:
        print(f"\n  Normalized Word Accuracy:")
        print(f"    Perfect words: {row['perfect_words']}/{row['total_words']} ({row['perfect_words']/max(row['total_words'],1):.1%})")
        print(f"    ≤1 char diff: {row['normalized_acc_1']:.1%}")
        print(f"    ≤2 char diff: {row['normalized_acc_2']:.1%}")

    # Full text comparison
    print("\n" + "=" * 80)
    print("FULL TEXT COMPARISON:")
    print("=" * 80)
    print(f"\nGROUND TRUTH ({len(row['ground_truth'])} chars):")
    print(row["ground_truth"])
    print(f"\nPREDICTION ({len(row['prediction'])} chars):")
    print(row["prediction"])

    # Character-level differences
    print(highlight_differences(row["prediction"], row["ground_truth"]))

    # Word-level differences
    print(show_word_differences(row["prediction"], row["ground_truth"]))

    # Show image if available
    if show_image and image_dir:
        img_path = image_dir / f"{row['id']}.jpg"
        if img_path.exists():
            print(f"\nOpening image: {img_path}")
            img = Image.open(img_path)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Sample: {row['id']}")
            plt.tight_layout()
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Inspect validation errors interactively")
    parser.add_argument("--results", required=True, help="Path to validation_error_analysis.csv")
    parser.add_argument("--image-dir", default="dataset/images", help="Path to image directory")
    parser.add_argument("--filter-by", choices=["wer", "cer", "score"], default="wer",
                       help="Metric to filter by")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top errors to show")
    parser.add_argument("--condition", help="Filter by condition tier (excellent/medium/poor/very_poor)")
    parser.add_argument("--text-type", help="Filter by text type (plain/names_only/has_caret/has_digits)")
    parser.add_argument("--show-images", action="store_true", help="Display images (requires matplotlib)")
    parser.add_argument("--sample-id", help="Inspect specific sample by ID")
    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        raise ValueError(f"Results file not found: {results_path}")

    print(f"Loading results from {results_path}...")
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} samples")

    # Apply filters
    filtered_df = df.copy()

    if args.condition:
        if "condition_tier" not in df.columns:
            print("Warning: condition_tier not in results, ignoring filter")
        else:
            filtered_df = filtered_df[filtered_df["condition_tier"] == args.condition]
            print(f"Filtered to condition={args.condition}: {len(filtered_df)} samples")

    if args.text_type:
        if "text_type" not in df.columns:
            print("Warning: text_type not in results, ignoring filter")
        else:
            filtered_df = filtered_df[filtered_df["text_type"] == args.text_type]
            print(f"Filtered to text_type={args.text_type}: {len(filtered_df)} samples")

    # Inspect specific sample or top errors
    image_dir = Path(args.image_dir)

    if args.sample_id:
        # Inspect specific sample
        sample = filtered_df[filtered_df["id"] == args.sample_id]
        if len(sample) == 0:
            print(f"Error: Sample ID '{args.sample_id}' not found")
            return

        inspect_sample(sample.iloc[0], image_dir, args.show_images)
    else:
        # Show top N errors
        metric_col = args.filter_by
        if metric_col == "score":
            metric_col = "competition_score"

        top_errors = filtered_df.nlargest(args.top_n, metric_col)

        print(f"\n{'=' * 80}")
        print(f"TOP {args.top_n} ERRORS BY {args.filter_by.upper()}")
        print(f"{'=' * 80}")

        for i, (_, row) in enumerate(top_errors.iterrows(), 1):
            print(f"\n{i}. {row['id']}: "
                  f"CER={row['cer']:.4f}, WER={row['wer']:.4f}, "
                  f"Score={row['competition_score']:.4f}")

            if "condition_tier" in row and pd.notna(row["condition_tier"]):
                print(f"   Condition: {row['condition_tier']}, ", end="")
            if "text_type" in row and pd.notna(row["text_type"]):
                print(f"Text type: {row['text_type']}, ", end="")
            if "text_length" in row and pd.notna(row["text_length"]):
                print(f"Length: {row['text_length']} chars")

        # Offer interactive inspection
        while True:
            print("\n" + "=" * 80)
            choice = input(f"\nEnter sample number (1-{len(top_errors)}) to inspect, "
                          f"or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                break

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(top_errors):
                    inspect_sample(top_errors.iloc[idx], image_dir, args.show_images)
                else:
                    print(f"Error: Please enter a number between 1 and {len(top_errors)}")
            except ValueError:
                print("Error: Please enter a valid number or 'q'")


if __name__ == "__main__":
    main()

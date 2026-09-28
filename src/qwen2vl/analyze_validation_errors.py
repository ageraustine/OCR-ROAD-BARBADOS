"""
Validation Error Analysis for Qwen-VL HTR Model

Generates detailed error analysis to understand where WER/CER comes from:
- Per-sample predictions with error breakdown
- Categorization by document characteristics (condition, text type, length)
- Error pattern analysis (character substitutions, word errors, etc.)
- Normalized word accuracy (near-miss words with edit distance ≤1, ≤2)

Usage:
    python analyze_validation_errors.py --checkpoint /path/to/checkpoint --config config.yaml
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor
from train import (
    resolve_model_class,
    levenshtein,
    word_levenshtein,
    corpus_cer,
    corpus_wer,
    REPO_ROOT,
    OCR_PROMPT,
)
from data_utils import (
    load_and_prepare_dataframe,
    make_splits,
    classify_caret_types,
    compute_text_type,
)


def load_image(path: str, max_pixels: int = 2016000) -> Image.Image:
    """Load and downscale image while preserving aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)."""
    return text.lower().strip()


def compute_character_error_types(pred: str, ref: str) -> dict:
    """
    Compute character-level error types (substitution, deletion, insertion).

    Returns dict with counts for each error type.
    """
    # Simple alignment-free approximation
    # For production, use proper alignment (e.g., Needleman-Wunsch)
    errors = {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }

    # Use Levenshtein to get total edits
    total_edits = levenshtein(pred, ref)

    # Simple heuristic: if pred is shorter, likely deletions; if longer, insertions
    len_diff = len(pred) - len(ref)

    if len_diff > 0:
        # Prediction is longer - likely insertions
        errors["insertions"] = min(len_diff, total_edits)
        errors["substitutions"] = total_edits - errors["insertions"]
    elif len_diff < 0:
        # Prediction is shorter - likely deletions
        errors["deletions"] = min(-len_diff, total_edits)
        errors["substitutions"] = total_edits - errors["deletions"]
    else:
        # Same length - all substitutions
        errors["substitutions"] = total_edits

    return errors


def compute_word_level_metrics(pred: str, ref: str) -> dict:
    """
    Compute word-level metrics including:
    - WER (standard word error rate)
    - Normalized word accuracy (words with edit distance ≤1, ≤2)
    - Word error types (substitution, deletion, insertion)
    """
    pred_words = pred.split()
    ref_words = ref.split()

    metrics = {
        "wer": 0.0,
        "word_edits": 0,
        "total_words": len(ref_words),
        "normalized_acc_1": 0.0,  # % words with edit distance ≤1
        "normalized_acc_2": 0.0,  # % words with edit distance ≤2
        "perfect_words": 0,
        "near_miss_1": 0,  # Words with 1 char difference
        "near_miss_2": 0,  # Words with 2 char differences
    }

    if len(ref_words) == 0:
        return metrics

    # Compute WER
    word_edits = word_levenshtein(pred_words, ref_words)
    metrics["word_edits"] = word_edits
    metrics["wer"] = word_edits / len(ref_words)

    # Compute normalized word accuracy
    # For simplicity, align by position (assumes mostly correct order)
    # For production, use proper word alignment
    perfect = 0
    near_1 = 0
    near_2 = 0

    for i, ref_word in enumerate(ref_words):
        if i >= len(pred_words):
            continue  # Deletion - already counted in WER

        pred_word = pred_words[i]
        char_dist = levenshtein(pred_word, ref_word)

        if char_dist == 0:
            perfect += 1
            near_1 += 1
            near_2 += 1
        elif char_dist == 1:
            near_1 += 1
            near_2 += 1
        elif char_dist == 2:
            near_2 += 1

    metrics["perfect_words"] = perfect
    metrics["near_miss_1"] = near_1 - perfect
    metrics["near_miss_2"] = near_2 - near_1
    metrics["normalized_acc_1"] = near_1 / len(ref_words)
    metrics["normalized_acc_2"] = near_2 / len(ref_words)

    return metrics


def categorize_sample(row: pd.Series, df: pd.DataFrame) -> dict:
    """
    Categorize a sample by its characteristics.

    Returns dict with:
    - condition_tier: good/medium/poor/very_poor (if condition_score available)
    - text_type: plain/names_only/has_caret/has_digits
    - length_tier: short/medium/long
    - has_interlineation: bool
    - has_superscript: bool
    """
    categories = {}

    # Condition tier (VLM-based v2 thresholds)
    if "condition_score" in df.columns and not pd.isna(row.get("condition_score")):
        score = row["condition_score"]
        if score < 20.5:
            categories["condition_tier"] = "excellent"
        elif score < 24.5:
            categories["condition_tier"] = "medium"
        elif score < 27.5:
            categories["condition_tier"] = "poor"
        else:
            categories["condition_tier"] = "very_poor"
    else:
        categories["condition_tier"] = "unknown"

    # Text type
    has_digit = bool(re.search(r"\d", str(row["Target"])))
    has_upper = bool(re.search(r"[A-Z]", str(row["Target"])))
    has_interlin, has_superscr = classify_caret_types(pd.Series([row["Target"]]))
    has_caret = has_interlin.iloc[0] or has_superscr.iloc[0]

    if has_digit:
        categories["text_type"] = "has_digits"
    elif has_caret:
        categories["text_type"] = "has_caret"
    elif has_upper:
        categories["text_type"] = "names_only"
    else:
        categories["text_type"] = "plain"

    categories["has_interlineation"] = has_interlin.iloc[0]
    categories["has_superscript"] = has_superscr.iloc[0]

    # Length tier (tertiles)
    text_len = len(str(row["Target"]))
    length_tertiles = df["Target"].str.len().quantile([0.33, 0.67]).values

    if text_len < length_tertiles[0]:
        categories["length_tier"] = "short"
    elif text_len < length_tertiles[1]:
        categories["length_tier"] = "medium"
    else:
        categories["length_tier"] = "long"

    categories["text_length"] = text_len

    return categories


@torch.no_grad()
def generate_predictions(model, processor, val_df, image_dir, max_pixels,
                        max_new_tokens=256, batch_size=4, num_beams=5):
    """
    Generate predictions for all validation samples.

    Returns list of dicts with:
    - id: sample ID
    - ground_truth: reference text
    - prediction: model output
    - cer: character error rate
    - wer: word error rate
    - char_errors: character-level error breakdown
    - word_metrics: word-level metrics
    """
    model.eval()
    tokenizer = processor.tokenizer
    results = []

    # Set up for batched generation
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for i in tqdm(range(0, len(val_df), batch_size), desc="Generating predictions"):
            chunk = val_df.iloc[i:i + batch_size]

            # Load images
            images = []
            for _, row in chunk.iterrows():
                img_path = image_dir / f"{row['ID']}.jpg"
                if not img_path.exists():
                    print(f"Warning: Image not found: {img_path}")
                    continue
                images.append(load_image(str(img_path), max_pixels))

            if not images:
                continue

            # Prepare prompts
            prompts = [
                processor.apply_chat_template(
                    [{"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": OCR_PROMPT},
                    ]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for img in images
            ]

            # Encode
            inputs = processor(
                text=prompts,
                images=images,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode
            trimmed = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)

            # Compute metrics
            for j, (_, row) in enumerate(chunk.iterrows()):
                if j >= len(decoded):
                    continue

                pred = decoded[j].strip()
                ref = str(row["Target"]).strip()

                # Character-level metrics
                cer = levenshtein(pred, ref) / max(len(ref), 1)
                char_errors = compute_character_error_types(pred, ref)

                # Word-level metrics
                word_metrics = compute_word_level_metrics(pred, ref)

                # Combined competition score
                score = 0.5 * cer + 0.5 * word_metrics["wer"]

                results.append({
                    "id": row["ID"],
                    "ground_truth": ref,
                    "prediction": pred,
                    "cer": cer,
                    "wer": word_metrics["wer"],
                    "competition_score": score,
                    "char_edits": levenshtein(pred, ref),
                    "word_edits": word_metrics["word_edits"],
                    "total_chars": len(ref),
                    "total_words": word_metrics["total_words"],
                    "char_errors": char_errors,
                    "perfect_words": word_metrics["perfect_words"],
                    "near_miss_1": word_metrics["near_miss_1"],
                    "near_miss_2": word_metrics["near_miss_2"],
                    "normalized_acc_1": word_metrics["normalized_acc_1"],
                    "normalized_acc_2": word_metrics["normalized_acc_2"],
                })

    finally:
        tokenizer.padding_side = prev_side

    return results


def analyze_error_patterns(results_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """
    Analyze error patterns across different categories.

    Returns dict with breakdowns by:
    - condition_tier
    - text_type
    - length_tier
    - caret_type
    """
    analysis = {
        "overall": {
            "mean_cer": results_df["cer"].mean(),
            "mean_wer": results_df["wer"].mean(),
            "mean_score": results_df["competition_score"].mean(),
            "median_cer": results_df["cer"].median(),
            "median_wer": results_df["wer"].median(),
            "normalized_acc_1": results_df["normalized_acc_1"].mean(),
            "normalized_acc_2": results_df["normalized_acc_2"].mean(),
        },
        "by_category": {},
    }

    # Categorize each sample
    categories_list = []
    for _, row in val_df.iterrows():
        categories_list.append(categorize_sample(row, val_df))

    categories_df = pd.DataFrame(categories_list)
    results_with_categories = pd.concat([results_df.reset_index(drop=True),
                                         categories_df.reset_index(drop=True)], axis=1)

    # Analyze by condition tier
    if "condition_tier" in categories_df.columns:
        by_condition = {}
        for tier in ["excellent", "medium", "poor", "very_poor"]:
            subset = results_with_categories[results_with_categories["condition_tier"] == tier]
            if len(subset) > 0:
                by_condition[tier] = {
                    "count": len(subset),
                    "mean_cer": subset["cer"].mean(),
                    "mean_wer": subset["wer"].mean(),
                    "mean_score": subset["competition_score"].mean(),
                    "normalized_acc_1": subset["normalized_acc_1"].mean(),
                }
        analysis["by_category"]["condition_tier"] = by_condition

    # Analyze by text type
    by_text_type = {}
    for text_type in ["plain", "names_only", "has_caret", "has_digits"]:
        subset = results_with_categories[results_with_categories["text_type"] == text_type]
        if len(subset) > 0:
            by_text_type[text_type] = {
                "count": len(subset),
                "mean_cer": subset["cer"].mean(),
                "mean_wer": subset["wer"].mean(),
                "mean_score": subset["competition_score"].mean(),
                "normalized_acc_1": subset["normalized_acc_1"].mean(),
            }
    analysis["by_category"]["text_type"] = by_text_type

    # Analyze by length tier
    by_length = {}
    for length_tier in ["short", "medium", "long"]:
        subset = results_with_categories[results_with_categories["length_tier"] == length_tier]
        if len(subset) > 0:
            by_length[length_tier] = {
                "count": len(subset),
                "mean_cer": subset["cer"].mean(),
                "mean_wer": subset["wer"].mean(),
                "mean_score": subset["competition_score"].mean(),
                "normalized_acc_1": subset["normalized_acc_1"].mean(),
            }
    analysis["by_category"]["length_tier"] = by_length

    # Analyze caret sub-types
    by_caret = {}
    interlin_subset = results_with_categories[results_with_categories["has_interlineation"] == True]
    if len(interlin_subset) > 0:
        by_caret["interlineation"] = {
            "count": len(interlin_subset),
            "mean_cer": interlin_subset["cer"].mean(),
            "mean_wer": interlin_subset["wer"].mean(),
            "mean_score": interlin_subset["competition_score"].mean(),
        }

    superscr_subset = results_with_categories[results_with_categories["has_superscript"] == True]
    if len(superscr_subset) > 0:
        by_caret["superscript"] = {
            "count": len(superscr_subset),
            "mean_cer": superscr_subset["cer"].mean(),
            "mean_wer": superscr_subset["wer"].mean(),
            "mean_score": superscr_subset["competition_score"].mean(),
        }

    if by_caret:
        analysis["by_category"]["caret_type"] = by_caret

    return analysis, results_with_categories


def print_analysis_report(analysis: dict, results_df: pd.DataFrame):
    """Print formatted analysis report."""
    print("\n" + "=" * 80)
    print("VALIDATION ERROR ANALYSIS REPORT")
    print("=" * 80)

    # Overall metrics
    print("\nOVERALL METRICS:")
    print(f"  Samples: {len(results_df)}")
    print(f"  Mean CER: {analysis['overall']['mean_cer']:.4f}")
    print(f"  Mean WER: {analysis['overall']['mean_wer']:.4f}")
    print(f"  Mean Score (0.5*CER + 0.5*WER): {analysis['overall']['mean_score']:.4f}")
    print(f"  Median CER: {analysis['overall']['median_cer']:.4f}")
    print(f"  Median WER: {analysis['overall']['median_wer']:.4f}")
    print(f"\n  Normalized Word Accuracy:")
    print(f"    ≤1 char diff: {analysis['overall']['normalized_acc_1']:.2%}")
    print(f"    ≤2 char diff: {analysis['overall']['normalized_acc_2']:.2%}")

    # By category
    print("\n" + "-" * 80)
    print("BREAKDOWN BY CATEGORY:")
    print("-" * 80)

    for category, breakdown in analysis["by_category"].items():
        print(f"\n{category.upper().replace('_', ' ')}:")

        # Sort by mean_score (worst first)
        sorted_items = sorted(breakdown.items(),
                             key=lambda x: x[1].get("mean_score", 0),
                             reverse=True)

        for tier, metrics in sorted_items:
            print(f"  {tier:15s} (n={metrics['count']:3d}): "
                  f"CER={metrics['mean_cer']:.4f}, "
                  f"WER={metrics['mean_wer']:.4f}, "
                  f"Score={metrics['mean_score']:.4f}")
            if "normalized_acc_1" in metrics:
                print(f"                           "
                      f"NW_acc(≤1)={metrics['normalized_acc_1']:.2%}")

    # Top errors
    print("\n" + "-" * 80)
    print("TOP 20 ERRORS BY WER:")
    print("-" * 80)

    top_wer = results_df.nlargest(20, "wer")
    for i, (_, row) in enumerate(top_wer.iterrows(), 1):
        print(f"\n{i}. ID: {row['id']} | WER={row['wer']:.3f}, CER={row['cer']:.3f}")
        print(f"   GT:   {row['ground_truth'][:100]}")
        print(f"   PRED: {row['prediction'][:100]}")
        if len(row['ground_truth']) > 100 or len(row['prediction']) > 100:
            print(f"   [truncated]")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze validation errors")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--output", default="validation_error_analysis.json",
                       help="Output JSON file for detailed results")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    model_name = cfg["model"]["name"]
    model_class, family = resolve_model_class(model_name)

    # Load model
    model = model_class.from_pretrained(
        str(checkpoint_path),
        device_map={"": 0} if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(str(checkpoint_path), trust_remote_code=True)

    # Load data
    print("\nLoading validation data...")
    df = load_and_prepare_dataframe(data_cfg)

    # Get validation split
    image_dir = REPO_ROOT / data_cfg["image_dir"]
    splits = list(make_splits(df, data_cfg))

    if len(splits) == 0:
        raise ValueError("No splits generated")

    # Use first split's validation set
    _, val_df, _ = splits[0]
    print(f"Validation set: {len(val_df)} samples")

    # Generate predictions
    print("\nGenerating predictions...")
    results = generate_predictions(
        model=model,
        processor=processor,
        val_df=val_df,
        image_dir=image_dir,
        max_pixels=train_cfg["max_pixels"],
        max_new_tokens=cfg.get("inference", {}).get("max_new_tokens", 256),
        batch_size=args.batch_size,
        num_beams=cfg.get("inference", {}).get("num_beams", 5),
    )

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Analyze error patterns
    print("\nAnalyzing error patterns...")
    analysis, results_with_categories = analyze_error_patterns(results_df, val_df)

    # Print report
    print_analysis_report(analysis, results_df)

    # Save detailed results
    output_path = REPO_ROOT / args.output

    # Convert to serializable format
    output_data = {
        "overall_metrics": analysis["overall"],
        "category_breakdown": analysis["by_category"],
        "per_sample_results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_path}")

    # Save CSV with categories for further analysis
    csv_path = output_path.with_suffix(".csv")
    results_with_categories.to_csv(csv_path, index=False)
    print(f"Per-sample CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

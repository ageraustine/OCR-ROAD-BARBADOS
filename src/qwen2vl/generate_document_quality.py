"""
VLM-based Document Quality Assessment
Generate reliable document_condition.csv using Qwen3-VL for historical HTR.

Analyzes each document image for:
- Physical damage (tears, holes, missing sections)
- Ink degradation (fading, bleeding, smudging)
- Paper condition (staining, discoloration, aging)
- Text readability (clarity, contrast, legibility)
- Overall quality composite score

Output: document_condition.csv with per-category scores + composite
"""

import os
import sys
import json
import re
import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for data_utils import
sys.path.append(str(Path(__file__).parent))
from data_utils import REPO_ROOT

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent

# VLM assessment prompt - structured to get reliable JSON scores
QUALITY_ASSESSMENT_PROMPT = """Analyze this historical handwritten document image and assess its condition.

Rate the following aspects on a scale of 0-100 (where 0=excellent/pristine and 100=severely degraded):

1. **physical_damage**: Tears, holes, missing sections, edge damage (0=no damage, 100=severe tears/holes)
2. **ink_degradation**: Ink fading, bleeding, smudging, or loss of contrast (0=dark clear ink, 100=barely visible)
3. **paper_condition**: Staining, discoloration, yellowing, mold, water damage (0=clean white, 100=heavily stained)
4. **text_readability**: Overall clarity and legibility of handwriting (0=very clear, 100=illegible)

Respond ONLY with a JSON object in this exact format:
{
    "physical_damage": <0-100>,
    "ink_degradation": <0-100>,
    "paper_condition": <0-100>,
    "text_readability": <0-100>
}"""


# Composite score weights (tunable based on what matters most for OCR)
DEFAULT_WEIGHTS = {
    "physical_damage": 0.20,      # Tears/holes affect some regions
    "ink_degradation": 0.40,      # Most critical for OCR (faded text hard to read)
    "paper_condition": 0.15,      # Background stains can interfere
    "text_readability": 0.25,     # Direct readability assessment
}


# ─────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────

def load_model_and_processor(model_name: str, max_pixels: int = 2016000):
    """Load Qwen3-VL model and processor for quality assessment."""
    print(f"Loading {model_name.split('/')[-1]}...", flush=True)

    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if has_flash_attention() else "sdpa",
    )
    model.eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        max_pixels=max_pixels,
    )

    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)")
    return model, processor


def has_flash_attention() -> bool:
    """Check if flash attention is available."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────
# QUALITY ASSESSMENT
# ─────────────────────────────────────────────────────────────

def assess_document_quality(
    image_path: Path,
    model,
    processor,
    max_pixels: int = 2016000,
    max_new_tokens: int = 256,
) -> Optional[Dict[str, float]]:
    """
    Assess document quality using VLM (single image).

    Returns:
        Dict with keys: physical_damage, ink_degradation, paper_condition,
        text_readability, or None if assessment failed.
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")

        # Downscale if needed (match training pipeline)
        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # Format prompt with chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": QUALITY_ASSESSMENT_PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate assessment
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic
                num_beams=1,
                temperature=None,
                top_p=None,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Decode response
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Parse JSON from response
        scores = parse_quality_response(response)

        return scores

    except Exception as e:
        print(f"  ⚠️  Error processing {image_path.name}: {e}")
        return None


def assess_document_quality_batch(
    image_paths: list[Path],
    model,
    processor,
    max_pixels: int = 2016000,
    max_new_tokens: int = 256,
    verbose: bool = False,
) -> list[Optional[Dict[str, float]]]:
    """
    Assess document quality using VLM (batched for efficiency).

    Args:
        image_paths: List of image paths to process
        model: VLM model
        processor: Processor
        max_pixels: Max image resolution
        max_new_tokens: Max tokens to generate
        verbose: Print raw VLM responses for debugging

    Returns:
        List of score dicts (one per image), None for failed assessments
    """
    try:
        # Load and preprocess all images
        images = []
        valid_indices = []

        for idx, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")

                # Downscale if needed
                w, h = img.size
                if w * h > max_pixels:
                    scale = (max_pixels / (w * h)) ** 0.5
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"  ⚠️  Error loading {img_path.name}: {e}")
                continue

        if not images:
            return [None] * len(image_paths)

        # Format prompts for batch
        messages_batch = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": QUALITY_ASSESSMENT_PROMPT},
                    ],
                }
            ]
            for img in images
        ]

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        # CRITICAL FIX: Set padding_side='left' for decoder-only batched generation
        original_padding_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = 'left'

        try:
            # Process batch
            inputs = processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # Generate assessments
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            # Decode responses
            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            responses = processor.batch_decode(generated_ids, skip_special_tokens=True)

        finally:
            # Restore original padding side
            processor.tokenizer.padding_side = original_padding_side

        # Parse all responses
        results = [None] * len(image_paths)
        for batch_idx, original_idx in enumerate(valid_indices):
            response = responses[batch_idx].strip()

            if verbose:
                print(f"\n--- Response for {image_paths[original_idx].name} ---")
                print(response)
                print("---\n")

            scores = parse_quality_response(response)

            if scores is None and verbose:
                print(f"  ⚠️  Failed to parse JSON from response for {image_paths[original_idx].name}")

            results[original_idx] = scores

        return results

    except Exception as e:
        print(f"  ⚠️  Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return None for all
        return [None] * len(image_paths)


def parse_quality_response(response: str) -> Optional[Dict[str, float]]:
    """
    Parse VLM response to extract quality scores.

    Handles various JSON formats and extracts scores robustly.
    """
    # Try multiple strategies to find JSON

    # Strategy 1: Find JSON object with proper nesting (handles multi-line)
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if not json_match:
        # Strategy 2: Try to extract just the outer braces
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

    if not json_match:
        return None

    json_str = json_match.group(0)

    # Clean up common formatting issues
    # Remove markdown code blocks if present
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*', '', json_str)

    try:
        data = json.loads(json_str)

        # Extract expected fields (try various naming conventions)
        required_fields = ["physical_damage", "ink_degradation", "paper_condition", "text_readability"]
        field_aliases = {
            "physical_damage": ["physical_damage", "physical", "damage", "tears_holes"],
            "ink_degradation": ["ink_degradation", "ink", "fading", "ink_quality"],
            "paper_condition": ["paper_condition", "paper", "staining", "paper_quality"],
            "text_readability": ["text_readability", "readability", "clarity", "legibility"],
        }

        scores = {}

        for field in required_fields:
            found = False
            # Try exact match first
            if field in data:
                try:
                    score = float(data[field])
                    scores[field] = max(0.0, min(100.0, score))
                    found = True
                except (ValueError, TypeError):
                    pass

            # Try aliases if exact match failed
            if not found:
                for alias in field_aliases.get(field, []):
                    if alias in data:
                        try:
                            score = float(data[alias])
                            scores[field] = max(0.0, min(100.0, score))
                            found = True
                            break
                        except (ValueError, TypeError):
                            continue

            if not found:
                # Missing field - return None (assessment failed)
                return None

        return scores

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None


def compute_composite_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Compute weighted composite quality score.

    Args:
        scores: Dict with physical_damage, ink_degradation, paper_condition, text_readability
        weights: Dict with same keys, values sum to 1.0

    Returns:
        Composite score in [0, 100] range (0=excellent, 100=very poor)
    """
    composite = sum(scores[key] * weights[key] for key in weights)
    return composite


# ─────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────

def process_dataset(
    image_dir: Path,
    image_ids: list,
    model,
    processor,
    output_csv: Path,
    resume: bool = True,
    max_pixels: int = 2016000,
    weights: Dict[str, float] = None,
    save_interval: int = 50,
    batch_size: int = 4,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Process entire dataset and generate quality scores.

    Args:
        image_dir: Directory containing images
        image_ids: List of image IDs to process
        model: Loaded VLM model
        processor: Loaded processor
        output_csv: Path to save results
        resume: If True, skip already processed images
        max_pixels: Max image resolution
        weights: Composite score weights (defaults to DEFAULT_WEIGHTS)
        save_interval: Save checkpoint every N images
        batch_size: Number of images to process per batch (default: 4)
        verbose: Print VLM responses for debugging

    Returns:
        DataFrame with quality scores
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Load existing results if resuming
    existing_results = {}
    if resume and output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)

            # Validate required columns
            required_cols = {"ID", "physical_damage", "ink_degradation", "paper_condition",
                           "text_readability", "composite_score", "success"}
            if not required_cols.issubset(existing_df.columns):
                missing = required_cols - set(existing_df.columns)
                print(f"⚠️  Existing CSV missing columns {missing} - starting fresh")
            else:
                # Only load successful assessments to avoid propagating placeholder scores
                valid_results = existing_df[existing_df["success"] == True].copy()
                existing_results = valid_results.set_index("ID").to_dict("index")

                n_total = len(existing_df)
                n_valid = len(valid_results)
                n_failed = n_total - n_valid

                print(f"Resuming: {n_valid} valid results loaded ({n_failed} failed assessments will be retried)")
        except Exception as e:
            print(f"Could not load existing results: {e} - starting fresh")

    # Process images in batches
    results = []
    failed_ids = []

    # Prepare batches of unprocessed images
    unprocessed = [(idx, img_id) for idx, img_id in enumerate(image_ids) if img_id not in existing_results]

    # Add already processed results
    for img_id in image_ids:
        if img_id in existing_results:
            results.append({"ID": img_id, **existing_results[img_id]})

    # Process in batches
    for batch_start in tqdm(range(0, len(unprocessed), batch_size), desc="Assessing quality (batched)"):
        batch_items = unprocessed[batch_start:batch_start + batch_size]
        batch_ids = [img_id for _, img_id in batch_items]
        batch_indices = [idx for idx, _ in batch_items]

        # Find image files for batch
        batch_paths = []
        valid_batch_items = []

        for (idx, img_id) in batch_items:
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                candidate = image_dir / f"{img_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                print(f"  ⚠️  Image not found: {img_id}")
                failed_ids.append(img_id)
            else:
                batch_paths.append(img_path)
                valid_batch_items.append((idx, img_id))

        if not batch_paths:
            continue

        # Assess quality for batch
        batch_scores = assess_document_quality_batch(
            batch_paths, model, processor, max_pixels=max_pixels, verbose=verbose
        )

        # Process results
        for (idx, img_id), scores in zip(valid_batch_items, batch_scores):
            # Track whether assessment succeeded BEFORE filling placeholders
            assessment_succeeded = scores is not None

            if scores is None:
                print(f"  ⚠️  Assessment failed: {img_id}")
                failed_ids.append(img_id)
                # Add placeholder scores (neutral) - these will NOT be used in training
                # because success=False filters them out in data_utils.py line 68
                scores = {
                    "physical_damage": 50.0,
                    "ink_degradation": 50.0,
                    "paper_condition": 50.0,
                    "text_readability": 50.0,
                }

            # Compute composite score
            composite = compute_composite_score(scores, weights)

            # Store result
            result = {
                "ID": img_id,
                "physical_damage": scores["physical_damage"],
                "ink_degradation": scores["ink_degradation"],
                "paper_condition": scores["paper_condition"],
                "text_readability": scores["text_readability"],
                "composite_score": composite,
                "success": assessment_succeeded,  # FIX: Use flag saved before placeholders
            }
            results.append(result)

        # Save checkpoint
        if len(results) % save_interval < batch_size:
            df = pd.DataFrame(results)
            # Validate before saving
            if len(df) > 0 and "success" in df.columns:
                # Create backup if file exists
                backup_path = output_csv.with_suffix('.csv.bak')
                if output_csv.exists():
                    output_csv.rename(backup_path)
                df.to_csv(output_csv, index=False)
                # Remove backup after successful save
                if output_csv.exists() and backup_path.exists():
                    backup_path.unlink()
                print(f"  Checkpoint saved: {len(results)} images processed ({df['success'].sum()} successful)")

    # Final save with backup
    df = pd.DataFrame(results)
    backup_path = output_csv.with_suffix('.csv.bak')
    if output_csv.exists():
        output_csv.rename(backup_path)
    df.to_csv(output_csv, index=False)
    # Remove backup after successful save
    if output_csv.exists() and backup_path.exists():
        backup_path.unlink()

    # Summary
    successful = sum(r["success"] for r in results)
    print(f"\n{'='*70}")
    print(f"Processing complete:")
    print(f"  Total: {len(image_ids)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(failed_ids)}")
    if failed_ids[:10]:
        print(f"  First failures: {', '.join(failed_ids[:10])}")
    print(f"  Saved: {output_csv}")
    print(f"{'='*70}\n")

    return df


# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────

def print_statistics(df: pd.DataFrame):
    """Print quality score statistics and stratification breakdown."""
    print(f"\n{'='*70}")
    print("QUALITY SCORE STATISTICS")
    print(f"{'='*70}")

    # Per-metric statistics
    metrics = ["physical_damage", "ink_degradation", "paper_condition", "text_readability", "composite_score"]

    for metric in metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean:   {values.mean():.2f}")
            print(f"  Median: {values.median():.2f}")
            print(f"  Std:    {values.std():.2f}")
            print(f"  Min:    {values.min():.2f}")
            print(f"  Max:    {values.max():.2f}")

    # Stratification breakdown (match training.py thresholds)
    if "composite_score" in df.columns:
        composite = df["composite_score"].dropna()

        print(f"\n{'='*70}")
        print("STRATIFICATION BREAKDOWN (Training Tiers)")
        print(f"{'='*70}")

        # Thresholds from train.py
        excellent = (composite < 8.07).sum()
        medium = ((composite >= 8.07) & (composite < 23.38)).sum()
        poor = ((composite >= 23.38) & (composite < 26.12)).sum()
        very_poor = (composite >= 26.12).sum()

        total = len(composite)

        print(f"Excellent (< 8.07):       {excellent:4d} ({100*excellent/total:5.1f}%)")
        print(f"Medium (8.07-23.38):      {medium:4d} ({100*medium/total:5.1f}%)")
        print(f"Poor (23.38-26.12):       {poor:4d} ({100*poor/total:5.1f}%)")
        print(f"Very Poor (>= 26.12):     {very_poor:4d} ({100*very_poor/total:5.1f}%)")
        print(f"{'='*70}\n")

        # Percentile breakdown
        print("Percentiles:")
        for p in [10, 25, 50, 75, 90, 95]:
            print(f"  {p:2d}th: {np.percentile(composite, p):.2f}")
        print()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate document quality scores using Qwen3-VL"
    )
    parser.add_argument(
        "--config",
        default="config_qwen3_8b.yaml",
        help="Config file (in configs/ directory)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Qwen3-VL model to use (default: 4B for speed)"
    )
    parser.add_argument(
        "--output",
        default="dataset/document_condition_v2.csv",
        help="Output CSV path relative to repo root (default: dataset/document_condition_v2.csv - VLM-based scores)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch (ignore existing results)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save checkpoint every N images"
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Override image directory from config"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Process only first N images (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images to process per batch (default: 4, increase for faster processing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print VLM responses for debugging failed assessments"
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / "configs" / args.config
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Determine image directory
    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        image_dir = REPO_ROOT / cfg["data"]["image_dir"]

    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)

    # Get list of all images
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG"))
    image_ids = sorted([f.stem for f in image_files])

    if args.subset:
        image_ids = image_ids[:args.subset]
        print(f"Processing subset: {args.subset} images")

    print(f"Found {len(image_ids)} images in {image_dir}")

    # Output path
    output_csv = REPO_ROOT / args.output

    # Load model
    max_pixels = cfg["training"].get("max_pixels", 2016000)
    model, processor = load_model_and_processor(args.model, max_pixels=max_pixels)

    # Process dataset
    print(f"\nProcessing {len(image_ids)} images...")
    print(f"Output: {output_csv}")
    print(f"Batch size: {args.batch_size}")
    print(f"Resume: {not args.no_resume}\n")

    df = process_dataset(
        image_dir=image_dir,
        image_ids=image_ids,
        model=model,
        processor=processor,
        output_csv=output_csv,
        resume=not args.no_resume,
        max_pixels=max_pixels,
        weights=DEFAULT_WEIGHTS,
        save_interval=args.save_interval,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # Print statistics
    print_statistics(df)

    print(f"\n✓ Document quality assessment complete!")
    print(f"  Results saved to: {output_csv}")
    print(f"  Use this file during training for adaptive augmentation\n")


if __name__ == "__main__":
    main()

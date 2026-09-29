# CLAUDE.md - R.O.A.D. Project Guide

## Project Overview

**Reclaiming Our Atlantic Destiny (R.O.A.D.)** - A Zindi ML competition for handwritten text recognition (HTR) on historical Barbados archival documents.

**Goal**: Build an OCR model to transcribe colonial-era handwritten text (deeds, wills, census records) from scanned images.

## Dataset Structure

```
dataset/
├── Train.csv          # 4098 samples (ID, Target)
├── Test.csv           # 1373 samples (ID only)
├── SampleSubmission.csv
└── images/            # 5472 JPG images
```

- **ID**: Image filename without extension (e.g., `uGI8F9Er0c5XwdnX`)
- **Target**: Ground truth transcription text

## Evaluation Metrics

Final score = 0.5 * WER + 0.5 * CER (lower is better)

- **WER**: Word Error Rate
- **CER**: Character Error Rate
- Longer transcriptions weighted more heavily

## Submission Format

```csv
ID,Target
MzQuRiUbPFsq6Azy,transcribed text here
```

## Starter Approaches

Located in `src/`:

| Approach | Directory | Best For | Training |
|----------|-----------|----------|----------|
| VLM | `VLM/` | Highest accuracy (Qwen2-VL) | Yes |
| Kraken-OCR | `Kraken-OCR/` | Historical documents | Yes |
| Paddle-OCR | `Paddle-OCR/` | Fast inference | Limited |

### Each starter contains:
- `setup.sh` - Environment setup (creates conda env)
- `config.yaml` - Configuration paths
- `inference.py` - Generate predictions
- `eval_metrics.py` - Calculate WER/CER
- `train.py` or `trainer.py` - Training script (VLM/Kraken)

## Image Analysis Tool

Before training, analyze your dataset to determine optimal image resolution settings:

```bash
# Analyze dataset/images to find optimal max_pixels
bash run_image_analysis.sh
```

This provides:
- Image dimension statistics (width, height, pixels)
- Resize impact at different thresholds
- Recommendations for config.yaml max_pixels setting
- Critical for balancing quality vs training speed

See `IMAGE_ANALYSIS_README.md` for details.

## Quick Start

```bash
# Example: VLM approach
cd src/VLM
bash setup.sh
conda activate vlm_env
# Edit config.yaml with correct paths
python trainer.py      # Train
python inference.py    # Generate submission.csv
python eval_metrics.py # Evaluate
```

## Key Challenges

- Faded ink and degraded pages
- Unfamiliar historical handwriting styles
- Variable text lengths

## System Requirements

- Python 3.11
- 16GB+ RAM (32GB recommended)
- CUDA GPU recommended (11.8+ for Kraken, 12.x for Paddle/VLM)

## Important Paths

- Images: `dataset/images/{ID}.jpg`
- Update `repo_root` and `base_image_dir` in config.yaml files

---

## Competition Pipeline (src/qwen2vl/)

**Current Best: 0.905 leaderboard score** (Qwen3-VL-8B with layer-wise LoRA)

### Architecture

- **Model**: Qwen3-VL-8B-Instruct (full precision, no quantization)
- **Fine-tuning**: Layer-wise LoRA strategy (vision-adapted)
  - **Vision early blocks (0-8)**: r=16 (preserve general features)
  - **Vision middle blocks (9-17)**: r=32 (domain adaptation)
  - **Vision late blocks (18-26)**: r=64 (OCR-specific features)
  - **Vision merger**: r=64 (critical vision→text bridge)
  - **LLM**: r=8 uniform (sufficient - 87.77% normalized word accuracy)
- **RSLoRA**: Enabled (rank-stabilized scaling for mixed-rank layers)
- **Flash Attention 2**: Optional (improves speed 20-30%, auto-fallback if unavailable)
- **Precision**: BF16 for optimal A100 utilization

### Key Features

- **Conservative augmentation**: CRITICAL - documents are already degraded
  - ✅ Brightness/contrast (scan variations)
  - ✅ Light rotation/shear (alignment variance only)
  - ❌ NO blur/noise/elastic (docs already have these!)
- **Simple prompt**: No examples or historical context (prevents hallucinations)
- **High-res processing**: 2M pixels (~1344x1500)
- **Auto-resume**: Training resumes from checkpoint if interrupted
- **Early stopping**: Monitors eval_score with patience=3

### Current Performance

**Validation metrics:**
- CER: 0.0467
- WER: 0.1498
- **Normalized word accuracy (≤1 char)**: 87.77% ← Excellent character recognition
- Combined eval_score: 0.098

**Leaderboard:** 0.905 (strong baseline)

**Key findings:**
- Vision r=64 is SUFFICIENT (87.77% accuracy proves excellent character recognition)
- LLM r=8 is SUFFICIENT (no capacity bottleneck)
- SHORT texts perform WORST (generation control issue, not capacity)

### Training Setup

**Step 0: Analyze Dataset (Recommended)**
```bash
bash run_image_analysis.sh
# Update max_pixels in config based on recommendations
```

**Option 1: Google Colab (Recommended)**
```
1. Open train_colab.ipynb
2. Set Runtime to GPU (A100)
3. Run all cells
```

**Option 2: Local/Remote GPU**
```bash
cd src/qwen2vl
pip install -r requirements.txt
python train.py

# Expected: ~6-8 hours on A100 80GB
# Auto-resumes from checkpoint if interrupted
```

### Robust Inference (Recommended Path Forward)

Since training has achieved strong baseline (0.905), further improvement comes from **inference-time optimization**:

```bash
cd src/qwen2vl

# Test all inference strategies on existing checkpoint
python robust_inference.py \
  --strategy all \
  --checkpoint /path/to/best/checkpoint \
  --batch-size 4

# Generates 7 submission files in inference_experiments/:
# - submission_greedy.csv (no beam search)
# - submission_beam_3.csv (conservative - less hallucination)
# - submission_beam_5.csv (standard - current baseline)
# - submission_short_text.csv (optimized for short texts)
# - submission_long_text.csv (handles underscores/repetition)
# - submission_sampling_low.csv (temperature=0.3)
# - submission_sampling_mid.csv (temperature=0.5)

# Then test ensemble methods
python robust_inference.py \
  --strategy ensemble \
  --checkpoint /path/to/best/checkpoint \
  --batch-size 4

# Generates 3 ensemble submissions:
# - submission_ensemble_voting.csv (word-level majority)
# - submission_ensemble_shortest.csv (prevents hallucinations)
# - submission_ensemble_longest.csv (handles underscore sequences)
```

**Why robust inference?**
- Validation (0.098) ≠ test distribution (0.905)
- Need to empirically test on actual leaderboard
- Training experiments (repetition_penalty, vision rank increases) haven't improved baseline

### Configuration (configs/config_qwen3_8b_vision_adapted.yaml)

**Training hyperparameters:**
- Batch size: 2 (effective 16 with grad_accum=8)
- Learning rate: 5e-5 with cosine schedule
- Epochs: 5
- Weight decay: 0.05
- Gradient checkpointing: Enabled

**LoRA configuration:**
```yaml
lora_r: 8  # Base rank for LLM
lora_alpha: 16
lora_dropout: 0.1
use_rslora: true  # CRITICAL for mixed-rank layers

# Layer-wise ranks via rank_pattern (see config for full regex)
# Vision: early=16, middle=32, late=64
# Merger: 64
# LLM: 8 (uniform)
```

**Augmentation (CONSERVATIVE - critical!):**
```yaml
p_blur: 0.0        # DISABLED - docs already blurred
p_noise: 0.0       # DISABLED - docs already noisy
p_elastic: 0.0     # DISABLED - pages already warped
p_brightness: 0.3  # Scan exposure variance
p_contrast: 0.3    # Ink fade variance
p_rotate: 0.3      # Reduced - alignment only
p_shear: 0.2       # Reduced - slant only
```

**Inference settings:**
```yaml
max_new_tokens: 150
num_beams: 3               # Reduced from 5 (less over-generation)
repetition_penalty: 1.0    # NO PENALTY (historical docs have legitimate repetition!)
length_penalty: 1.0        # Neutral
no_repeat_ngram_size: 0    # Allow exact repetition (for "___" and boilerplate)
```

**OCR Prompt (simplified):**
```
Transcribe the visible text exactly as written.
Preserve all spelling, capitalization, punctuation, and special characters.
Transcribe only what you see in the image—nothing more.
```

### What Doesn't Work (Documented Failures)

See `LESSONS_LEARNED.md` for details:

❌ **Repetition penalty > 1.0**: Made WER 20% worse (0.1618 → 0.1935)
  - Historical documents HAVE legitimate repetition ("the said", "___  ___")
  - Penalty forced model to use different words, increasing errors

❌ **Lexicon post-processing**: Dropped leaderboard score (0.905 → 0.890)
  - Training lexicon ≠ test distribution
  - Over-corrected already-correct predictions

❌ **Aggressive augmentation**: Adds degradation to already-degraded docs
  - blur/noise/elastic make training harder without benefit

❌ **Increasing vision rank beyond r=64**: Not needed
  - 87.77% normalized word accuracy proves vision is excellent

❌ **Increasing LLM rank beyond r=8**: Not needed
  - Character recognition already excellent

### What Works (Evidence-Based)

✅ **Conservative augmentation**: Only brightness/contrast/rotation (no blur/noise/elastic)
✅ **Simple prompt**: No examples or historical context
✅ **Vision r=64**: Sufficient capacity (proven by validation metrics)
✅ **LLM r=8**: Sufficient capacity (excellent character recognition)
✅ **Layer-wise LoRA**: Early=16, middle=32, late=64 (vision adaptation)
✅ **RSLoRA**: Keeps mixed-rank layer contributions balanced
✅ **Reduced beam search**: num_beams=3 (less over-generation than 5)

### Next Steps for Improvement

**Priority 1: Robust Inference Testing** ⭐⭐⭐
- Test all 7 inference strategies on leaderboard
- Identify which handles short texts better
- Try ensemble methods (voting/shortest/longest)
- Expected gain: Δ -0.005 to -0.015

**Priority 2: Test-Time Augmentation (TTA)**
- Multiple predictions per image (rotation, brightness variants)
- Ensemble via word-level voting
- Expected gain: Δ -0.005 to -0.010

**Priority 3: Multi-Model Ensemble**
- Train 2-3 models with different seeds
- Combine predictions
- Expected gain: Δ -0.005 to -0.010

**Don't do:**
- ❌ Increase model capacity (already sufficient)
- ❌ Add repetition penalty (breaks legitimate boilerplate)
- ❌ Post-processing with lexicons (distribution mismatch)
- ❌ More aggressive augmentation (hurts performance)

### Files Reference

- `train.py` - Main training script
- `inference.py` - Standard inference (single strategy)
- `robust_inference.py` - Multi-strategy inference testing
- `configs/config_qwen3_8b_vision_adapted.yaml` - Current best config
- `LESSONS_LEARNED.md` - Documented failed approaches
- `IMPROVEMENT_GUIDE.md` - Evidence-based improvement strategy

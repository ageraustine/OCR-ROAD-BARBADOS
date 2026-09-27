# Document Quality Assessment with Qwen3-VL

Generate reliable `document_condition_v2.csv` using VLM-based quality assessment for adaptive augmentation during training.

**Note**: This generates v2 (VLM-based) scores, which are more reliable than v1 (CV-based metrics). The training pipeline automatically uses v2 when available.

## Overview

The `generate_document_quality.py` script uses Qwen3-VL to analyze each historical document image and generate quality scores across multiple dimensions:

- **Physical Damage**: Tears, holes, missing sections (0=pristine, 100=severe)
- **Ink Degradation**: Fading, bleeding, smudging (0=dark/clear, 100=barely visible)
- **Paper Condition**: Staining, discoloration, aging (0=clean, 100=heavily damaged)
- **Text Readability**: Overall clarity and legibility (0=very clear, 100=illegible)
- **Composite Score**: Weighted combination of all metrics (used for training stratification)

## Quick Start

```bash
cd src/qwen2vl

# Full dataset (5,472 images, ~1-2 hours on A100 with batching)
python generate_document_quality.py

# Test on subset (10 images)
python generate_document_quality.py --subset 10

# Faster processing with larger batch size (requires more VRAM)
python generate_document_quality.py --batch-size 8

# Use 8B model (slower but more accurate)
python generate_document_quality.py --model Qwen/Qwen3-VL-8B-Instruct

# Specify custom output location (path relative to repo root)
python generate_document_quality.py --output dataset/my_custom_quality.csv
```

## Usage

### Basic Command
```bash
python generate_document_quality.py [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `config_qwen3_8b.yaml` | Config file (in configs/ directory) |
| `--model` | `Qwen/Qwen3-VL-4B-Instruct` | Model to use (4B=faster, 8B=more accurate) |
| `--output` | `dataset/document_condition_v2.csv` | Output CSV path (VLM-based v2 format) |
| `--batch-size` | 4 | Images per batch (increase for speed, decrease if OOM) |
| `--no-resume` | False | Start from scratch (ignore existing results) |
| `--save-interval` | 50 | Save checkpoint every N images |
| `--image-dir` | From config | Override image directory |
| `--subset` | None | Process only first N images (for testing) |

### Examples

**Test on 50 images:**
```bash
python generate_document_quality.py --subset 50
```

**Use 8B model (higher quality):**
```bash
python generate_document_quality.py --model Qwen/Qwen3-VL-8B-Instruct
```

**Faster processing with larger batch (A100 80GB):**
```bash
python generate_document_quality.py --batch-size 8  # 2x faster
python generate_document_quality.py --batch-size 16 # 3-4x faster (requires ~40GB VRAM)
```

**Smaller batch for limited VRAM (RTX 4090 24GB):**
```bash
python generate_document_quality.py --batch-size 2
```

**Start fresh (ignore previous results):**
```bash
python generate_document_quality.py --no-resume
```

**Save checkpoints every 100 images:**
```bash
python generate_document_quality.py --save-interval 100
```

## Output Format

### CSV Structure
```csv
ID,physical_damage,ink_degradation,paper_condition,text_readability,composite_score,success
abc123,15.0,35.0,25.0,20.0,26.5,True
def456,5.0,10.0,8.0,12.0,9.8,True
```

### Columns

- **ID**: Image filename (without extension)
- **physical_damage**: 0-100 score (0=no damage, 100=severe tears/holes)
- **ink_degradation**: 0-100 score (0=dark ink, 100=faded)
- **paper_condition**: 0-100 score (0=clean, 100=stained)
- **text_readability**: 0-100 score (0=clear, 100=illegible)
- **composite_score**: Weighted average (used for training stratification)
- **success**: True if VLM assessment succeeded, False if fallback used

### Composite Score Weights

Default weights (tunable in `DEFAULT_WEIGHTS`):

```python
{
    "physical_damage": 0.20,      # 20% - Localized impact
    "ink_degradation": 0.40,      # 40% - Most critical for OCR
    "paper_condition": 0.15,      # 15% - Background interference
    "text_readability": 0.25,     # 25% - Overall legibility
}
```

## Training Integration

The generated `document_condition_v2.csv` is automatically used by `train.py` for **adaptive augmentation**.

### How It Works

1. **Generate quality scores** (this script):
   ```bash
   python generate_document_quality.py
   ```

2. **Training loads scores** automatically:
   ```python
   # train.py line 731-735
   if has_condition:
       condition = row.get("condition_score")
       if condition is not None and not pd.isna(condition):
           sample["condition_score"] = float(condition)
   ```

3. **Augmentation adapts** based on condition (train.py line 331-453):
   - **Excellent (< 8.07)**: Aggressive augmentation + degradation synthesis
   - **Medium (8.07-23.38)**: Standard augmentation
   - **Poor (23.38-26.12)**: No degradation, 1.5× geometric diversity
   - **Very Poor (≥ 26.12)**: No degradation, 2.5× geometric diversity

### Automatic Integration

The training script (`data_utils.py` lines 63-81) automatically loads and merges VLM-based quality scores:

1. **Looks for** `dataset/document_condition_v2.csv` in repo root
2. **Filters** to successful assessments only (`success == True`)
3. **Merges** `composite_score` on `ID` column with `Train.csv` (renamed to `condition_score` internally)
4. **Fills** missing scores with median (for any failed assessments)

**No config changes needed** - just generate the CSV and the training pipeline will detect and use it automatically.

**Note**: The pipeline uses v2 (VLM-based) exclusively. If you have old v1 (CV-based) files, they will be ignored.

## Performance

### Speed Estimates (with batching)

| Model | GPU | Batch Size | Time (5,472 images) | Throughput |
|-------|-----|------------|---------------------|------------|
| Qwen3-VL-4B | A100 80GB | 4 (default) | ~1-2 hours | ~50-90 img/min |
| Qwen3-VL-4B | A100 80GB | 8 | ~40-60 min | ~90-140 img/min |
| Qwen3-VL-4B | A100 80GB | 16 | ~30-45 min | ~120-180 img/min |
| Qwen3-VL-8B | A100 80GB | 4 | ~2-3 hours | ~30-45 img/min |
| Qwen3-VL-4B | RTX 4090 24GB | 2 | ~2-3 hours | ~30-45 img/min |
| Qwen3-VL-4B | RTX 4090 24GB | 4 | ~1.5-2 hours | ~45-60 img/min |

**Batching speedup**: 2-4x faster than single-image processing

**VRAM usage**:
- Batch size 4: ~20-25GB (A100/RTX 4090)
- Batch size 8: ~35-40GB (A100 only)
- Batch size 16: ~50-60GB (A100 80GB only)

### Resume Capability

The script auto-saves every 50 images (configurable). If interrupted:

```bash
# Resume from where it left off
python generate_document_quality.py  # Automatically skips processed images
```

Force restart:
```bash
python generate_document_quality.py --no-resume
```

## Quality Stratification

### Expected Distribution

Based on historical document analysis, expect roughly:

- **Excellent**: ~20-25% (pristine/well-preserved documents)
- **Medium**: ~55-60% (typical aging/wear)
- **Poor**: ~10-12% (significant degradation)
- **Very Poor**: ~5-10% (severe damage/fading)

### Example Output Statistics

```
QUALITY SCORE STATISTICS
======================================================================

Composite Score:
  Mean:   18.45
  Median: 16.20
  Std:    12.30
  Min:    2.50
  Max:    78.90

STRATIFICATION BREAKDOWN (Training Tiers)
======================================================================
Excellent (< 8.07):       1254 (22.9%)
Medium (8.07-23.38):      3152 (57.6%)
Poor (23.38-26.12):        552 (10.1%)
Very Poor (>= 26.12):      514 ( 9.4%)
======================================================================
```

## Troubleshooting

### Issue: "Model not found"
**Solution**: Install latest transformers:
```bash
pip install transformers>=4.57.0
```

### Issue: "CUDA out of memory"
**Solution 1**: Reduce batch size:
```bash
python generate_document_quality.py --batch-size 2
python generate_document_quality.py --batch-size 1  # slowest but works on any GPU
```

**Solution 2**: Use 4B model instead of 8B:
```bash
python generate_document_quality.py --model Qwen/Qwen3-VL-4B-Instruct
```

### Issue: "JSON parsing failed"
**Solution**: The script handles this gracefully with fallback scores (50.0 for all metrics). Check the `success` column in output CSV to see which images failed.

### Issue: "Assessment failed for many images"
**Causes**:
- VLM not following JSON format (try 8B model for better instruction following)
- Images corrupted or unreadable
- Prompt needs adjustment

**Debug**:
```python
# Add print statement in assess_document_quality() to see raw VLM responses
print(f"Response: {response}")
```

## Advanced

### Custom Weights

Edit `DEFAULT_WEIGHTS` in the script to prioritize different aspects:

```python
# Example: Prioritize ink quality over physical damage
DEFAULT_WEIGHTS = {
    "physical_damage": 0.10,      # Reduced
    "ink_degradation": 0.50,      # Increased
    "paper_condition": 0.15,
    "text_readability": 0.25,
}
```

Regenerate:
```bash
python generate_document_quality.py --no-resume
```

### Optimal Batch Size

Choose based on your GPU and priorities:

**For maximum speed (A100 80GB):**
```bash
python generate_document_quality.py --batch-size 16  # ~30-45 min for full dataset
```

**For balanced speed/memory (RTX 4090 24GB, A100):**
```bash
python generate_document_quality.py --batch-size 4   # Default, good balance
```

**For limited VRAM (16GB GPUs):**
```bash
python generate_document_quality.py --batch-size 1   # Slowest but safest
```

**Rule of thumb**: Each batch slot adds ~8-10GB VRAM for 4B model, ~12-15GB for 8B model.

## Next Steps

1. **Generate VLM-based quality scores** (v2):
   ```bash
   python generate_document_quality.py
   ```
   This creates `dataset/document_condition_v2.csv` in your repo root with VLM-assessed quality metrics.

2. **Review statistics** to verify reasonable distribution:
   - Check the stratification breakdown
   - Ensure distribution matches expectations (see Expected Distribution above)
   - Identify any failed assessments

3. **Train with adaptive augmentation** (automatic):
   ```bash
   python train.py --config config_qwen3_8b.yaml
   ```
   The training script will automatically detect and load `dataset/document_condition_v2.csv`.
   You'll see this message during training:
   ```
   Loaded VLM-based document condition scores from document_condition_v2.csv
     Mean: 18.5, Median: 16.2, Std: 12.3
   ```

4. **Compare performance** vs baseline (no quality scores):
   - Rename the CSV temporarily to disable adaptive augmentation:
     ```bash
     mv dataset/document_condition_v2.csv dataset/document_condition_v2.csv.bak
     ```
   - Train without quality scores (baseline run)
   - Restore and train with adaptive augmentation:
     ```bash
     mv dataset/document_condition_v2.csv.bak dataset/document_condition_v2.csv
     ```
   - Compare metrics between adaptive and baseline runs
   - Check if poor-condition documents improve with adaptive augmentation
   - Verify no degradation on excellent-condition documents

## Troubleshooting Data Integration

If training doesn't load the quality scores:

1. **Check file location**:
   ```bash
   ls dataset/document_condition_v2.csv  # Should exist
   ```

2. **Verify CSV format** (VLM-based v2):
   ```bash
   head -5 dataset/document_condition_v2.csv
   # Should have columns: ID,physical_damage,ink_degradation,paper_condition,text_readability,composite_score,success
   # Verify composite_score column exists (v2 format)
   ```

3. **Check training output** for this line:
   ```
   Loaded VLM-based document condition scores from document_condition_v2.csv
   ```
   If you see "Document condition scores not found (document_condition_v2.csv)" instead, the file is missing or in the wrong location.

## Citation

If using this quality assessment in research:

```bibtex
@software{road_quality_assessment,
  title={VLM-based Document Quality Assessment for Historical HTR},
  author={R.O.A.D. Competition Team},
  year={2026},
  note={Qwen3-VL quality scoring for adaptive augmentation}
}
```

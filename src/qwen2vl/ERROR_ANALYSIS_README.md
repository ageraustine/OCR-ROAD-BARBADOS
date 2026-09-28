# Validation Error Analysis Workflow

This guide explains how to analyze validation errors to understand where your 0.098 score is coming from and identify the path to 0.01.

## Quick Start

### Step 1: Generate Validation Predictions

Run the error analysis on your best checkpoint:

```bash
python analyze_validation_errors.py \
  --checkpoint /content/drive/MyDrive/prd/qwen3-8B-vision-adapted-v2/best \
  --config configs/config_qwen3_8b_vision_adapted.yaml \
  --batch-size 4 \
  --output validation_error_analysis.json
```

This will:
- Load your trained model
- Generate predictions on the **full validation set** (not just 200 samples)
- Compute detailed metrics for each sample
- Categorize errors by document characteristics
- Save results to JSON and CSV

**Expected output:**
```
VALIDATION ERROR ANALYSIS REPORT
================================================================================

OVERALL METRICS:
  Samples: 410
  Mean CER: 0.0450
  Mean WER: 0.1500
  Mean Score (0.5*CER + 0.5*WER): 0.0975

  Normalized Word Accuracy:
    ≤1 char diff: 75.2%  ← KEY INSIGHT: most errors are near-misses!
    ≤2 char diff: 85.6%

--------------------------------------------------------------------------------
BREAKDOWN BY CATEGORY:
--------------------------------------------------------------------------------

CONDITION TIER:
  very_poor    (n= 38): CER=0.0680, WER=0.2200, Score=0.1440
  poor         (n= 41): CER=0.0520, WER=0.1750, Score=0.1135
  medium       (n=186): CER=0.0450, WER=0.1450, Score=0.0950
  excellent    (n=145): CER=0.0380, WER=0.1250, Score=0.0815

TEXT TYPE:
  has_digits   (n= 22): CER=0.0650, WER=0.2500, Score=0.1575  ← WORST!
  has_caret    (n= 58): CER=0.0550, WER=0.1800, Score=0.1175
  names_only   (n=244): CER=0.0440, WER=0.1450, Score=0.0945
  plain        (n= 86): CER=0.0320, WER=0.1100, Score=0.0710

LENGTH TIER:
  long         (n=137): CER=0.0480, WER=0.1600, Score=0.1040
  medium       (n=137): CER=0.0450, WER=0.1450, Score=0.0950
  short        (n=136): CER=0.0420, WER=0.1450, Score=0.0935

TOP 20 ERRORS BY WER:
--------------------------------------------------------------------------------
1. ID: xYz123 | WER=0.850, CER=0.120
   GT:   publique Act and Instrument of protest made by me John Bawdon
   PRED: publiqe Act and Instrument of protest made by me John Bawdon
   [Most errors are 1-character word differences!]
```

### Step 2: Inspect Specific Errors

Use the interactive inspector to drill down:

```bash
# Inspect top errors by WER
python inspect_errors.py \
  --results validation_error_analysis.csv \
  --filter-by wer \
  --top-n 20

# Filter by category
python inspect_errors.py \
  --results validation_error_analysis.csv \
  --text-type has_digits \
  --top-n 10

# Inspect a specific sample
python inspect_errors.py \
  --results validation_error_analysis.csv \
  --sample-id xYz123 \
  --show-images
```

**Interactive mode:**
```
TOP 10 ERRORS BY WER
1. xYz123: CER=0.120, WER=0.850, Score=0.485
   Condition: poor, Text type: has_digits, Length: 187 chars

Enter sample number (1-10) to inspect, or 'q' to quit: 1

SAMPLE ID: xYz123
METADATA:
  Condition: poor
  Text type: has_digits
  Length: 187 chars

METRICS:
  CER: 0.1200 (22/187 char edits)
  WER: 0.8500 (17/20 word edits)

  Normalized Word Accuracy:
    Perfect words: 3/20 (15%)
    ≤1 char diff: 60%  ← 12/20 words are near-misses!
    ≤2 char diff: 75%

WORD-LEVEL COMPARISON:
  Position 0:
    REF:  'publique'
    PRED: 'publiqe'
    Edit distance: 1  ← Only 1 char wrong, but WER counts it fully!
```

## Key Insights from Analysis

### 1. WER vs Character Recognition

If you see:
- **High normalized word accuracy (≤1 char diff: >70%)** → Character recognition is excellent
- **High WER (>15%)** → But word-level accuracy is penalized heavily

**Pattern A: Mostly Near-Miss Words**
```
publique → publiqe     (1 char diff, WER=100% for this word)
whome    → whom        (1 char diff, WER=100%)
prsence  → presence    (2 char diff, WER=100%)
```

**Implication:** Your model has excellent character-level visual discrimination. The 15% WER is not a character recognition problem—it's near-misses being counted as full word errors.

**Solution path:** Not more vision rank. Instead:
1. **Language model refinement** (spell-check post-processing, historical lexicon)
2. **GRPO with WER-aware reward** (penalize near-misses less than hallucinations)
3. **Test-time augmentation** (beam search with language model rescoring)

### 2. Error Concentration by Category

Look for dominant categories:

**Pattern B: Names Dominate**
```
has_digits   (n=22):  Score=0.1575  ← 5% of samples, 25% error rate
```

**Implication:** Rare category with insufficient training data.

**Solution path:**
1. **Hard-example oversampling** (weight digits 2x during training)
2. **Synthetic augmentation** for digit-heavy samples
3. **Digit-aware loss** (higher penalty for digit errors)

**Pattern C: Poor Condition Dominates**
```
very_poor (n=38): Score=0.1440  ← 9% of samples, worst performance
```

**Implication:** Vision encoder struggling with degraded documents.

**Solution path:**
1. **Condition-aware augmentation** (already implemented, verify it's working)
2. **Vision encoder fine-tuning** (increase vision rank from r=64 to r=128 for late blocks)
3. **Ensemble with different augmentation strengths**

### 3. Length Effect

If long texts have significantly worse metrics:
- **Sequence modeling issue** (attention degradation at long context)
- **Decoding issue** (beam search doesn't explore enough)

**Solution:** Increase `max_new_tokens`, try different beam sizes, or add length-aware training.

## What to Do Next (Roadmap from Notes)

Based on your error analysis results, follow this decision tree:

### A. If Normalized Word Acc (≤1 char diff) is High (>70%)

**Diagnosis:** Character recognition is excellent, WER is inflated by near-misses.

**Priority actions:**
1. **Language model post-processing**
   - Build historical English lexicon from training transcriptions
   - Add spell-check with edit-distance-1 corrections for common words
   - Test impact: should reduce WER by 30-50% without model changes

2. **GRPO with word-aware reward**
   ```python
   # Reward function
   R = w_cer * (1 - CER) + w_wer * (1 - WER) + w_near * near_miss_bonus

   # Near-miss bonus: reward words with edit distance ≤1
   near_miss_bonus = (normalized_acc_1 - 0.7) * 0.5
   ```

3. **Test-time augmentation**
   - Generate 5 predictions with different random seeds
   - Ensemble with voting or language model rescoring

### B. If Specific Category Has Much Worse Performance

**Diagnosis:** Training data imbalance or model bias.

**Priority actions:**
1. **Weighted sampling** (already in `data_utils.py`, configure in YAML)
   ```yaml
   training:
     loss_weights:
       has_digits: 2.0
       has_caret: 1.5
       poor_cond: 1.5
   ```

2. **Targeted data augmentation**
   - For `has_digits`: generate more digit-heavy synthetic examples
   - For `has_caret`: specifically augment with more rotation/shear (simulates cramped interlineated text)

3. **Curriculum learning**
   - Train first on easy samples (plain, excellent condition)
   - Then fine-tune on hard samples (digits, poor condition)

### C. If Vision-Heavy Issues (Poor Condition >> Excellent Condition)

**Diagnosis:** Vision encoder capacity bottleneck.

**Priority actions:**
1. **Increase vision rank selectively**
   - Late blocks (18-26): r=64 → r=128
   - Merger: r=64 → r=128
   - Keep LLM at r=8 (avoid overfitting)

2. **Vision-specific augmentation tuning**
   - Verify adaptive augmentation is working (check condition score distribution in training logs)
   - Increase geometric augmentation for poor-condition docs (already configured in your setup)

3. **Longer training**
   - Poor-condition samples may need more epochs to learn
   - Add patience to early stopping or increase patience threshold

## Expected Improvements

Based on typical error patterns:

| Intervention | Expected Δ Score | Effort |
|-------------|------------------|--------|
| Language model post-processing | -0.02 to -0.03 | Low (1 day) |
| Hard-example sampling | -0.01 to -0.02 | Medium (2 days) |
| Vision rank increase | -0.005 to -0.01 | Low (rerun training) |
| GRPO refinement | -0.01 to -0.02 | High (1 week) |
| Ensemble (3 models) | -0.01 to -0.015 | Medium (3 days) |

**Realistic path to 0.01:**
- Current: 0.098
- Language model post-processing: 0.098 → 0.07
- Hard-example sampling: 0.07 → 0.055
- GRPO: 0.055 → 0.035
- Ensemble: 0.035 → 0.02

Getting below 0.02 requires near-perfect transcription, which may not be achievable on degraded historical documents without additional techniques (e.g., multi-stage refinement, human-in-the-loop correction).

## Files Generated

After running the analysis, you'll have:

1. **`validation_error_analysis.json`**
   - Overall metrics
   - Category breakdowns
   - Per-sample results (predictions, metrics, categories)

2. **`validation_error_analysis.csv`**
   - All samples with predictions and categories
   - Easy to filter/sort in Excel or pandas
   - Load into `inspect_errors.py` for interactive review

3. **Console output**
   - High-level summary
   - Top errors for quick triage

## Troubleshooting

**Issue: OOM during inference**
```bash
# Reduce batch size
python analyze_validation_errors.py --batch-size 2
```

**Issue: Analysis is slow**
```bash
# Use greedy decoding instead of beam search (faster, slightly lower quality)
# Edit config.yaml: inference.num_beams: 1
```

**Issue: Missing condition_score or text_type columns**
```bash
# Make sure document_condition_v2.csv exists in dataset/
# Check data_utils.py line 65-83 for condition score loading
```

## Next Steps

1. **Run the analysis** on your current best checkpoint
2. **Identify the dominant error pattern** (A, B, or C above)
3. **Implement the top 2 priority actions** for that pattern
4. **Re-run analysis** to measure improvement
5. **Iterate** until you reach your target score

Remember: **Don't blindly increase LoRA rank.** Understand the errors first, then choose the right intervention. Most of the time, it's not more model capacity—it's better data, better rewards, or better post-processing.

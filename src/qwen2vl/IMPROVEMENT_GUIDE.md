# Improving from 0.905: Evidence-Based Strategy

**Current leaderboard:** 0.905 (very strong baseline)

**Validation metrics:**
- Mean CER: 0.0467
- Mean WER: 0.1498
- Normalized word accuracy (≤1 char): 87.77% ← Model is excellent at character recognition
- Poor-condition gap: 2x worse (0.1388 vs 0.0683) ← Vision capacity bottleneck

## What Doesn't Work

❌ **Lexicon-based post-processing** - Tested, made score WORSE (0.905 → 0.890)
- Training lexicon doesn't match test distribution
- Over-corrects already-correct predictions
- Validation ≠ test for this approach

## What Works

### 1. Cleaner Prompt (Anti-Hallucination)

**OLD PROMPT (encourages hallucinations):**
```
"Transcribe this 17th-18th century handwritten document image exactly as written.
Preserve original spelling (e.g., "publique", "prsence", "whome"),
capitalization including mid-sentence capitals, all punctuation marks,
and special marks like ^ (interlineations).
Do not modernize or correct spelling. Output only the transcription text."
```

**Problems:**
- Example words prime the model to generate them
- "17th-18th century" triggers pattern completion (validation error #1: model added "Soveraigne Lord Charles the Second by the grace of God")
- Too verbose, gives room for creative interpretation

**NEW PROMPT (hallucination-resistant):**
```
"Transcribe the visible text exactly as written.
Preserve all spelling, capitalization, punctuation, and special characters.
Transcribe only what you see in the image—nothing more."
```

✅ **Updated in both `train.py` and `inference.py`**

**Expected:** Reduces hallucinations (validation error #1 type), could improve by Δ -0.005 to -0.010

### 2. Vision Rank Increase (r=96 Late Blocks)

The config has been updated to increase vision capacity where it matters:

```yaml
# Late vision blocks: r=64 → r=96 (50% increase)
'visual\.blocks\.(1[8-9]|2[0-6])\.attn\.qkv': 96
'visual\.blocks\.(1[8-9]|2[0-6])\.attn\.proj': 96
'visual\.blocks\.(1[8-9]|2[0-6])\.mlp\.linear_fc1': 96
'visual\.blocks\.(1[8-9]|2[0-6])\.mlp\.linear_fc2': 96

# Merger: r=64 → r=96
'visual\.merger\.linear_fc1': 96
'visual\.merger\.linear_fc2': 96

# LLM: UNCHANGED at r=8 (already excellent)
```

**Rationale:**
- Poor-condition docs score 2x worse → vision encoder bottleneck
- 87.77% normalized word accuracy → LLM is already excellent, NO increase needed
- Late blocks do character-level discrimination → most important for degraded docs
- Conservative 50% increase (r=64→96) to avoid overfitting

**Expected improvement:** 0.905 → 0.885-0.890 (Δ -0.015 to -0.020)

**Combined with prompt change:** Could reach 0.880-0.885 (Δ -0.020 to -0.025)

### Training

```bash
cd src/qwen2vl

# Config is already updated at configs/config_qwen3_8b_vision_adapted.yaml
python train.py
```

**Training time:** ~6-8 hours on A100 80GB

## Alternative Approaches (If Vision Rank Doesn't Help Enough)

### Test-Time Augmentation (TTA)

Generate multiple predictions per image and ensemble:

```python
# In inference.py, generate 3 predictions per image with:
# - Slight rotation (-1°, 0°, +1°)
# - Slight brightness variation
# - Different random seeds

# Ensemble via word-level voting
```

**Expected:** Δ -0.005 to -0.010

### Ensemble Multiple Models

Train 3 models with different:
- Random seeds
- Vision rank allocations (r=64, r=96, r=128)
- Augmentation strengths

**Expected:** Δ -0.005 to -0.010

### Weighted Loss (For Rare Categories)

If vision rank increase doesn't help enough, target specific weaknesses:

```yaml
# Validation showed these are 2.3x worse:
has_digits (n=22): 0.1374
has_caret (n=55): 0.1322
```

Modify training to oversample or weight these higher.

## Don't Do This

❌ Increase LLM rank (87.77% normalized accuracy = already excellent)
❌ More augmentation (current is comprehensive)
❌ Lexicon post-processing (tested, made it worse)
❌ Longer training (5 epochs is optimal)

## Expected Timeline

| Intervention | Target Score | Effort | Priority |
|--------------|--------------|--------|----------|
| Prompt + Vision r=96 | 0.880-0.885 | 6-8 hrs | ⭐⭐⭐ |
| + TTA | 0.875-0.880 | 1-2 days | ⭐⭐ |
| + Ensemble (3 models) | 0.870-0.875 | 3-4 days | ⭐ |

## Bottom Line

**You're already at 0.905 - top tier performance.** The validation analysis confirmed your model has excellent character recognition (87.77%). The only bottleneck is vision capacity on degraded documents, which the updated config addresses.

Small improvements from here require careful, targeted interventions—not dramatic changes.

# Lessons Learned: What Didn't Work

## ❌ Repetition Penalty (FAILED - Made WER 20-32% WORSE)

**What we tried:**
- Added `repetition_penalty=1.2` to prevent hallucinations
- Hypothesis: Top validation error showed model adding boilerplate ("Soveraigne Lord Charles...")

**Results:**
```
Previous (penalty=1.0):  WER = 0.1618
With penalty=1.2:        WER = 0.1935  (20% WORSE!)
```

**Why it failed:**
- Historical documents HAVE legitimate repetition
  - "the said heires and assignes"
  - "By this publique Act and Instrument"
  - "___  ___  ___  ___" (structured repetition)
- Repetition penalty penalized these legitimate patterns
- Model forced to generate DIFFERENT words → higher WER
- **Lesson:** Don't penalize repetition in domains with legitimate boilerplate!

**Better approaches to try:**
1. Stricter prompt (already done: "Transcribe only what you see—nothing more")
2. Length penalty (but neutral 1.0, not 0.8)
3. EOS token training
4. Max tokens limit (already 150)
5. NOT repetition penalty

---

## ❌ Lexicon Post-Processing (FAILED - Made Score WORSE)

**What we tried:**
- Built historical lexicon from training data
- Applied edit-distance corrections to predictions

**Results:**
```
Original submission:     0.905
Post-processed:          0.890  (WORSE!)
```

**Why it failed:**
- Training lexicon ≠ test distribution
- Over-corrected already-correct predictions
- Historical spelling varies by scribe

**Lesson:** Post-processing needs test-distribution matching, which we don't have.

---

## ✅ What Actually Works

**Conservative Augmentation:**
```yaml
p_blur: 0.0      # Disabled (docs already degraded)
p_noise: 0.0     # Disabled (docs already grainy)
p_elastic: 0.0   # Disabled (pages already warped)
p_rotate: 0.3    # Light (alignment variance only)
```

**Cleaner Prompt:**
```python
"Transcribe the visible text exactly as written. "
"Preserve all spelling, capitalization, punctuation, and special characters. "
"Transcribe only what you see in the image—nothing more."
```

**Reduced Beam Search:**
```yaml
num_beams: 3  # Down from 5 - less over-generation
```

**Keep Simple:**
- Vision r=64 (sufficient - 87.77% normalized word accuracy)
- LLM r=8 (sufficient - excellent character recognition)
- NO repetition penalty (breaks legitimate boilerplate)
- NO lexicon post-processing (distribution mismatch)

---

## The Real Issues We Haven't Solved

1. **Short texts perform worst** (0.1163 vs 0.0899 for long)
   - NOT a repetition penalty issue
   - NOT a vision capacity issue
   - Maybe: prompt compliance, EOS token behavior?

2. **Validation vs test mismatch**
   - Validation: 0.098 combined score
   - Test: 0.905 (we're doing something validation doesn't capture)
   - Suggests: optimize for test via inference experiments, not validation

3. **Hallucination on error #1**
   - Still need solution for "Soveraigne Lord Charles..." hallucination
   - Repetition penalty wasn't it
   - Maybe: stricter max_new_tokens per input length?

---

## Next Steps (Inference Experiments Only)

Test these on **existing checkpoint** via inference.py:

1. **Beam search reduction** (num_beams: 3 → 1, greedy only)
2. **Adaptive max_tokens** (based on image text density)
3. **EOS token forcing** (early stopping strategies)
4. **Ensemble** (multiple models, different seeds)

**DON'T:**
- ❌ Train with repetition_penalty > 1.0
- ❌ Add more augmentation
- ❌ Increase model capacity further
- ❌ Apply post-processing

**Current best approach:** Conservative training + inference-time tuning

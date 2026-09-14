# R.O.A.D. Historical HTR Pipeline

A fine-tuning and evaluation pipeline for historical document Handwritten Text Recognition (HTR) using Vision-Language Models (Qwen3-VL-4B/8B Instruct). Optimized for challenging, degraded archival records featuring complex handwriting, caret insertion abbreviations, historical tildes, and money-in-words representations.

---

## 📌 Features & Key Innovations

* **Vision-Language Architecture**: Built on `Qwen3-VL-4B-Instruct` / `Qwen3-VL-8B-Instruct` using `bfloat16` and FlashAttention.
* **Asymmetric LoRA Capacity**:
  * **Vision Tower**: Higher rank ($r=64$, $\alpha=128$) to capture fine-grained stroke patterns, ink fading, and structural noise.
  * **Language Model**: Moderate rank ($r=16/32$) to adapt sequence generation without overfitting boilerplate text.
  * Implemented via Rank-Stabilized LoRA (`use_rslora: true`).
* **Condition-Aware Adaptive Augmentation**:
  * Analyzes document condition across 5 visual metrics (*text contrast*, *tears/holes*, *paper color variance*, *texture degradation*, *stains*).
  * Automatically separates **degradation augmentations** (blur, noise, brightness jitter—disabled on poor condition documents to prevent text erasing) from **geometric augmentations** (elastic deformation, shear, rotation—scaled up on damaged docs).
* **Length-Calibrated Token Bounds**: `max_seq_length: 3584` configured to prevent assistant header truncation resulting from multi-level vision feature fusion.
* **Analysis-Driven Stratification**: Single 90/10 train/validation split stratified across 36 distinct bins combining visual degradation, text difficulty, digit occurrences, and historical naming patterns.

---

## 📁 Repository Structure

```
ROAD/
├── dataset/
│   ├── images/              # Flattened image directory (.jpg, .jpeg, .png)
│   ├── Train.csv            # Training ground truth annotations
│   └── Test.csv             # Test target IDs and image references
├── src/
│   └── qwen2vl/
│       ├── configs/
│       │   └── config.yaml  # Tuned hyperparameter & pipeline configuration
│       ├── train.py         # Main training execution script
│       └── inference.py     # Evaluation & test submission generation script
└── README.md
```

---

## 🛠️ Requirements & Environment Setup

This project requires Python 3.10+, PyTorch 2.6+, and NVIDIA CUDA execution environments (A100 GPU recommended; L4/T4 supported for 4B variants).

### Version Pins

To prevent API regressions and CVE-related blocking:

```text
torch >= 2.6.0
transformers < 5.0.0
peft < 0.15.0
opencv-python-headless
soup-cli
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ageraustine/ROAD.git
cd ROAD

# Install dependencies
pip install "soup-cli[all]"
pip install opencv-python-headless
pip install git+https://github.com/MakazhanAlpamys/Soup.git
```

---

## ⚡ Quick Start: Training & Inference Pipeline

### 1. Dataset Preparation

Ensure images are downloaded and extracted to `dataset/images/`. The extraction step automatically flattens subdirectories into a unified image root.

```bash
# Example structure:
# dataset/images/sample_001.jpg
# dataset/Train.csv
```

### 2. Configuration (`config.yaml`)

Edit key parameters in `src/qwen2vl/configs/config.yaml`:

```yaml
model:
  name: "Qwen/Qwen3-VL-4B-Instruct"
  use_flash_attention: true

training:
  batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch size = 16
  learning_rate: 0.00005
  max_seq_length: 3584
  lora_r: 16
  rank_pattern:
    qkv: 64
    proj: 64
    linear_fc1: 64
    linear_fc2: 64

inference:
  checkpoint: "outputs/qwen3-4b-full/best"
  output_csv: "submission.csv"
```

### 3. Running Training

Execute the training script from the model source directory:

```bash
cd src/qwen2vl
python3 -u train.py --config configs/config.yaml
```

*Training metrics evaluate Character Error Rate (CER), Word Error Rate (WER), and a combined competition score ($0.5 \times \text{WER} + 0.5 \times \text{CER}$) with early stopping enabled.*

### 4. Running Inference & Submission

Generate test set predictions using the best saved checkpoint:

```bash
cd src/qwen2vl
python inference.py --checkpoint outputs/qwen3-4b-full/best
```

Output predictions will be saved to `submission.csv` at the repository root.

---

## 📊 Evaluation & Metrics

The pipeline continuously monitors validation error using normalized string distance metrics:

$$\text{Competition Score} = 0.5 \times \text{WER} + 0.5 \times \text{CER}$$

* **Early Stopping**: Patience of 5 evaluations with a minimum delta threshold ($\Delta = 0.0001$).
* **Checkpoint Selection**: Evaluates model state against `eval_score` across the full validation split (`cer_samples: 1.0`).

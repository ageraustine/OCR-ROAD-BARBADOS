"""
Qwen2-VL Inference for Historical HTR
Generates submission.csv for competition

Usage:
  python inference.py --config configs/config_qwen3_8b.yaml
  python inference.py --checkpoint outputs/qwen3-8b-full/best/

AUTO-CONFIG: No config needed after training!
  python inference.py
  → Automatically finds and uses summary.json from most recent training
"""

import os
import json
import argparse
import warnings
from pathlib import Path

import yaml
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor
from peft import PeftModel

# Suppress repetitive padding side warnings during batched inference
warnings.filterwarnings(
    "ignore",
    message=".*padding_side.*",
    category=UserWarning,
    module="transformers"
)

# Support both Qwen2-VL and Qwen3-VL
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

try:
    from transformers import Qwen2VLForConditionalGeneration
    QWEN2_AVAILABLE = True
except ImportError:
    QWEN2_AVAILABLE = False

# Fallback to generic class
if not QWEN3_AVAILABLE and not QWEN2_AVAILABLE:
    from transformers import AutoModelForVision2Seq

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent

OCR_PROMPT = (
    "Transcribe the visible text exactly as written. "
    "Preserve all spelling, capitalization, punctuation, and special characters. "
    "Transcribe only what you see in the image—nothing more."
)


# ─────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────

def load_image(path: str, max_pixels: int = 2016000) -> Image.Image:
    """Load and resize image while preserving aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size

    pixels = w * h
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


def clean_output(text: str) -> str:
    """Clean model output, removing genuine chat-template leakage only."""
    text = str(text)

    # FIXED (2026-08-31): the previous artifact list was ["<|assistant|>",
    # "<|user|>", "assistant", "Assistant:"] - none of the first two are
    # real Qwen tokens (Qwen's ChatML format is "<|im_start|>role\n...
    # <|im_end|>" - role names are plain text after <|im_start|>, never
    # wrapped in their own pipe-delimiters; verified directly against
    # train.py's own ASSISTANT_HEADER = "<|im_start|>assistant\n" constant).
    # So the old list provided ZERO actual protection against real Qwen
    # chat-template leakage, while the bare "assistant"/"Assistant:" entries
    # actively corrupted legitimate content: text.split(artifact)[-1] deletes
    # everything up to and including the first match, and "assistant" is an
    # ordinary English word ("with the assistance of", "assistant to the
    # said notary") that plausibly appears in genuine 17th-18th century
    # legal prose. Demonstrated directly: "John Smith assistant to the said
    # notary publique" was silently mangled into " to the said notary
    # publique" - a correct transcription destroyed by post-processing,
    # not a real artifact.
    #
    # Replaced with the actual Qwen ChatML markers. These are still matched
    # as substrings (not full-token-boundary-safe) but are FAR less likely
    # to occur in genuine transcribed text than a bare English word - "<|"
    # is not a sequence historical manuscripts produce.
    artifacts = ["<|im_start|>assistant", "<|im_start|>user", "<|im_start|>system", "<|im_end|>"]
    for artifact in artifacts:
        if artifact in text:
            # Keep content BEFORE the marker, not after. The realistic
            # failure mode is the model finishing its real answer correctly
            # and then failing to stop cleanly, hallucinating a new turn
            # afterward - we want to keep the genuine answer and discard the
            # hallucinated continuation. split()[-1] (keep-after) was tried
            # first and caught during verification: it turned
            # "By this publique Act<|im_end|>" into "" (empty), destroying a
            # complete, correct transcription just because it ended cleanly.
            text = text.split(artifact)[0]

    # REMOVED (2026-08-31, per explicit request): previously called
    # remove_repetitions(text, max_repetitions=5) here as a "defensive
    # layer" against degenerate generation loops. Dropped in favor of
    # trusting the model's raw output - this task's own domain has
    # legitimate repetition (legal boilerplate, repeated separator marks),
    # and repetition_penalty=1.0 was already deliberately chosen for the
    # same reason at generation time. A second, heuristic truncation layer
    # after generation risked cutting real content for the same reason the
    # artifact-stripping above did, even though remove_repetitions' specific
    # thresholds looked reasonably conservative on inspection. If degenerate
    # repetition turns out to be a real problem in practice, address it via
    # generation-time controls (e.g. a repetition_penalty > 1.0, or a
    # no_repeat_ngram_size) rather than post-hoc text surgery, so the
    # decision is visible in generation parameters rather than silently
    # applied after the fact.

    return " ".join(text.split()).strip()


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, model_name: str, use_flash: bool = True):
    """Load fine-tuned model with LoRA weights."""

    # Detect model type
    is_qwen3 = "Qwen3" in model_name or "qwen3" in model_name.lower()
    is_qwen2 = "Qwen2" in model_name or "qwen2" in model_name.lower()

    # Setup kwargs based on model type
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # Qwen3 uses dtype="auto", Qwen2 uses torch_dtype=torch.bfloat16
    if is_qwen3:
        model_kwargs["dtype"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    # Select appropriate model class
    if is_qwen3 and QWEN3_AVAILABLE:
        model_class = Qwen3VLForConditionalGeneration
    elif is_qwen2 and QWEN2_AVAILABLE:
        model_class = Qwen2VLForConditionalGeneration
    else:
        model_class = AutoModelForVision2Seq

    if use_flash:
        try:
            # Try with Flash Attention 2
            model_kwargs["attn_implementation"] = "flash_attention_2"
            base_model = model_class.from_pretrained(
                model_name,
                **model_kwargs,
            )
        except Exception:
            # Fallback to standard attention
            model_kwargs.pop("attn_implementation", None)
            base_model = model_class.from_pretrained(
                model_name,
                **model_kwargs,
            )
    else:
        base_model = model_class.from_pretrained(
            model_name,
            **model_kwargs,
        )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Set left padding for decoder-only model (required for correct batched generation)
    processor.tokenizer.padding_side = 'left'

    return model, processor


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_single(
    model,
    processor,
    image: Image.Image,
    max_new_tokens: int = 256,
    num_beams: int = 5,
    repetition_penalty: float = 1.0,
) -> str:
    """Generate transcription for a single image."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]

    # Qwen3 style: tokenize=True in apply_chat_template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )

    # Trim input prompt from generated output (Qwen3 style)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode only the generated part
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return clean_output(output_text[0] if output_text else "")


def predict_batch(
    model,
    processor,
    images: list[Image.Image],
    max_new_tokens: int = 256,
    num_beams: int = 5,
    repetition_penalty: float = 1.0,
) -> list[str]:
    """
    Generate transcriptions for a batch of images.

    Args:
        model: The model to use for generation
        processor: The processor for tokenization
        images: List of PIL images to process
        max_new_tokens: Maximum tokens to generate
        num_beams: Number of beams for beam search
        repetition_penalty: Penalty for token repetition (>1.0 = less repetition)

    Returns:
        List of transcription strings (same order as input images)
    """
    if not images:
        return []

    # Build messages for each image
    messages_batch = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ]
        for img in images
    ]

    # Apply chat template to each conversation
    text_inputs = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]

    # Process batch (processor handles padding automatically)
    inputs = processor(
        text=text_inputs,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )

    # Trim input prompt from generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode all outputs
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Clean each output
    return [clean_output(text) for text in output_texts]


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────

def run_inference(cfg: dict, checkpoint_override: str = None):
    """Run single-model inference on test set and generate submission with batching."""

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    inf_cfg = cfg["inference"]

    # Paths
    test_csv = REPO_ROOT / data_cfg["test_csv"]
    image_dir = REPO_ROOT / data_cfg["image_dir"]

    if checkpoint_override:
        checkpoint = Path(checkpoint_override)
    else:
        checkpoint = REPO_ROOT / inf_cfg["checkpoint"]

    output_csv = REPO_ROOT / inf_cfg["output_csv"]

    # Load model
    print(f"Loading model...", end=" ", flush=True)
    model, processor = load_model(
        str(checkpoint),
        model_cfg["name"],
        use_flash=model_cfg.get("use_flash_attention", True),
    )
    print("✓")

    # Load test data
    df = pd.read_csv(test_csv)
    batch_size = inf_cfg.get("batch_size", 2)
    print(f"Running inference: {len(df)} samples (batch_size={batch_size})")

    # Run predictions in batches
    results = []
    batch_ids = []
    batch_images = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting",
                      mininterval=2.0, ncols=80):
        img_id = str(row["ID"]).strip()
        img_path = image_dir / f"{img_id}.jpg"

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            results.append({"ID": img_id, "Target": ""})
            continue

        image = load_image(str(img_path))
        batch_ids.append(img_id)
        batch_images.append(image)

        # Process batch when full
        if len(batch_images) >= batch_size:
            preds = predict_batch(
                model,
                processor,
                batch_images,
                max_new_tokens=inf_cfg["max_new_tokens"],
                num_beams=inf_cfg["num_beams"],
                repetition_penalty=inf_cfg.get("repetition_penalty", 1.0),
            )
            for bid, pred in zip(batch_ids, preds):
                results.append({"ID": bid, "Target": pred})

            batch_ids = []
            batch_images = []

    # Process remaining images
    if batch_images:
        preds = predict_batch(
            model,
            processor,
            batch_images,
            max_new_tokens=inf_cfg["max_new_tokens"],
            num_beams=inf_cfg["num_beams"],
            repetition_penalty=inf_cfg.get("repetition_penalty", 1.0),
        )
        for bid, pred in zip(batch_ids, preds):
            results.append({"ID": bid, "Target": pred})

    # Save submission
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")

    return out_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2-VL HTR Inference - Generate competition submission"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file (auto-searches for summary.json if not found)")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--output", default=None, help="Override output CSV path")
    args = parser.parse_args()

    # Load config (with automatic fallback to summary.json)
    config_path = SCRIPT_DIR / args.config

    if config_path.exists():
        # Use provided config file
        print(f"Loading config: {config_path.name}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        # Try to find summary.json from most recent training
        print(f"Config not found: {config_path}")
        print("Searching for summary.json from training...")

        # Look for summary.json in common output directories
        summary_candidates = []

        # Check default output_dir patterns
        if (REPO_ROOT / "outputs").exists():
            summary_candidates.extend((REPO_ROOT / "outputs").rglob("summary.json"))

        # Check Google Drive paths (common for Colab)
        drive_paths = [
            Path("/content/drive/MyDrive/prd"),
            REPO_ROOT / "outputs",
        ]
        for drive_path in drive_paths:
            if drive_path.exists():
                summary_candidates.extend(drive_path.rglob("summary.json"))

        # Sort by modification time (most recent first)
        summary_candidates = sorted(
            [s for s in summary_candidates if s.exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if summary_candidates:
            summary_path = summary_candidates[0]
            print(f"✓ Found training summary: {summary_path}")
            print(f"  (Last modified: {pd.Timestamp.fromtimestamp(summary_path.stat().st_mtime)})")

            with open(summary_path) as f:
                summary = json.load(f)

            # BUG FIX: train.py writes summary.json as {"config": {...}, "results": [...]}
            # (see train.py's train(): json.dump({"config": cfg, "results": results}, f, ...)).
            # The actual model/data/training/inference config is nested under "config" -
            # using the raw summary dict directly as cfg meant cfg["model"] etc. always
            # raised KeyError, since the top-level keys were only "config" and "results".
            if "config" not in summary:
                raise KeyError(
                    f"{summary_path} has no 'config' key - this doesn't look like a "
                    f"summary.json written by train.py's train(). Found top-level keys: "
                    f"{list(summary.keys())}"
                )
            cfg = summary["config"]

            print(f"  Using config from training run")
        else:
            raise FileNotFoundError(
                f"Config not found: {config_path}\n"
                f"Also searched for summary.json but none found.\n"
                f"Run training first or provide valid --config path."
            )

    # CLI overrides
    if args.output:
        cfg["inference"]["output_csv"] = args.output

    # Single model inference
    if args.checkpoint:
        print(f"Using custom checkpoint: {args.checkpoint}")
    run_inference(cfg, checkpoint_override=args.checkpoint)


if __name__ == "__main__":
    main()
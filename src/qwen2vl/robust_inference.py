"""
Robust Inference for HTR - Multiple Strategies

Test different inference configurations to find optimal parameters
for the test distribution (not validation).

Usage:
    # Test single strategy
    python robust_inference.py --strategy greedy --checkpoint /path/to/checkpoint

    # Test all strategies
    python robust_inference.py --strategy all --checkpoint /path/to/checkpoint

    # Ensemble best strategies
    python robust_inference.py --strategy ensemble --checkpoint /path/to/checkpoint
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import yaml
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor
from peft import PeftModel

# Model imports
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

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Updated prompt
OCR_PROMPT = (
    "Transcribe the visible text exactly as written. "
    "Preserve all spelling, capitalization, punctuation, and special characters. "
    "Transcribe only what you see in the image—nothing more."
)


def load_image(path: str, max_pixels: int = 2016000) -> Image.Image:
    """Load and optionally resize image."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


class InferenceStrategy:
    """Base class for inference strategies."""

    def __init__(self, name: str):
        self.name = name

    def get_generation_config(self) -> dict:
        """Return generation config for this strategy."""
        raise NotImplementedError

    def __str__(self):
        return self.name


class GreedyStrategy(InferenceStrategy):
    """Fast greedy decoding - no beam search."""

    def __init__(self):
        super().__init__("greedy")

    def get_generation_config(self) -> dict:
        return {
            "max_new_tokens": 150,
            "do_sample": False,
            "num_beams": 1,  # Greedy
            "repetition_penalty": 1.0,
            "eos_token_id": None,  # Set by caller
            "pad_token_id": None,  # Set by caller
        }


class ConservativeBeamStrategy(InferenceStrategy):
    """Conservative beam search - less hallucination."""

    def __init__(self):
        super().__init__("beam_3")

    def get_generation_config(self) -> dict:
        return {
            "max_new_tokens": 150,
            "do_sample": False,
            "num_beams": 3,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "early_stopping": True,
            "eos_token_id": None,
            "pad_token_id": None,
        }


class StandardBeamStrategy(InferenceStrategy):
    """Standard beam search - current baseline."""

    def __init__(self):
        super().__init__("beam_5")

    def get_generation_config(self) -> dict:
        return {
            "max_new_tokens": 150,
            "do_sample": False,
            "num_beams": 5,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "early_stopping": True,
            "eos_token_id": None,
            "pad_token_id": None,
        }


class ShortTextStrategy(InferenceStrategy):
    """Optimized for short texts - prevent over-generation."""

    def __init__(self):
        super().__init__("short_text")

    def get_generation_config(self) -> dict:
        return {
            "max_new_tokens": 100,  # Stricter limit
            "do_sample": False,
            "num_beams": 1,  # Greedy to avoid exploration
            "repetition_penalty": 1.0,
            "eos_token_id": None,
            "pad_token_id": None,
        }


class LongTextStrategy(InferenceStrategy):
    """Optimized for long texts with repetition (underscores)."""

    def __init__(self):
        super().__init__("long_text")

    def get_generation_config(self) -> dict:
        return {
            "max_new_tokens": 200,  # More tokens
            "do_sample": False,
            "num_beams": 3,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,  # Allow repetition
            "eos_token_id": None,
            "pad_token_id": None,
        }


class SamplingStrategy(InferenceStrategy):
    """Sampling with temperature - diversity."""

    def __init__(self, temperature: float = 0.3):
        super().__init__(f"sampling_t{temperature}")
        self.temperature = temperature

    def get_generation_config(self) -> dict:
        return {
            "max_new_tokens": 150,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "eos_token_id": None,
            "pad_token_id": None,
        }


STRATEGIES = {
    "greedy": GreedyStrategy(),
    "beam_3": ConservativeBeamStrategy(),
    "beam_5": StandardBeamStrategy(),
    "short_text": ShortTextStrategy(),
    "long_text": LongTextStrategy(),
    "sampling_low": SamplingStrategy(0.3),
    "sampling_mid": SamplingStrategy(0.5),
}


@torch.no_grad()
def run_inference(model, processor, test_df, image_dir, strategy, batch_size=4):
    """
    Run inference with given strategy.

    Returns:
        List of predictions
    """
    model.eval()
    tokenizer = processor.tokenizer
    predictions = []

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        gen_config = strategy.get_generation_config()
        gen_config["eos_token_id"] = tokenizer.eos_token_id
        gen_config["pad_token_id"] = tokenizer.pad_token_id

        print(f"\nRunning strategy: {strategy.name}")
        print(f"  Config: {gen_config}")

        for i in tqdm(range(0, len(test_df), batch_size), desc=f"{strategy.name}"):
            chunk = test_df.iloc[i:i + batch_size]

            # Load images
            images = []
            for _, row in chunk.iterrows():
                img_path = image_dir / f"{row['ID']}.jpg"
                if not img_path.exists():
                    print(f"Warning: Missing {img_path}")
                    continue
                images.append(load_image(str(img_path)))

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
            outputs = model.generate(**inputs, **gen_config)

            # Decode
            trimmed = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)

            predictions.extend([d.strip() for d in decoded])

    finally:
        tokenizer.padding_side = prev_side

    return predictions


def ensemble_predictions(predictions_list: list, method: str = "voting") -> list:
    """
    Ensemble multiple prediction sets.

    Args:
        predictions_list: List of prediction lists (one per strategy)
        method: "voting" (word-level majority) or "longest" or "shortest"

    Returns:
        Ensembled predictions
    """
    n_samples = len(predictions_list[0])
    ensembled = []

    for i in range(n_samples):
        preds_for_sample = [preds[i] for preds in predictions_list]

        if method == "longest":
            # Choose longest prediction (for underscore sequences)
            ensembled.append(max(preds_for_sample, key=len))
        elif method == "shortest":
            # Choose shortest prediction (avoid hallucinations)
            ensembled.append(min(preds_for_sample, key=len))
        elif method == "voting":
            # Word-level majority voting
            all_words = [p.split() for p in preds_for_sample]
            max_len = max(len(words) for words in all_words)

            voted_words = []
            for pos in range(max_len):
                # Get words at this position from each prediction
                words_at_pos = [
                    words[pos] for words in all_words if pos < len(words)
                ]
                if words_at_pos:
                    # Majority vote
                    word_counts = Counter(words_at_pos)
                    voted_word = word_counts.most_common(1)[0][0]
                    voted_words.append(voted_word)

            ensembled.append(" ".join(voted_words))
        else:
            # Default: use first prediction
            ensembled.append(preds_for_sample[0])

    return ensembled


def main():
    parser = argparse.ArgumentParser(description="Robust inference testing")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--strategy", default="all",
                       choices=list(STRATEGIES.keys()) + ["all", "ensemble"],
                       help="Inference strategy to use")
    parser.add_argument("--test-csv", default="dataset/Test.csv")
    parser.add_argument("--image-dir", default="dataset/images")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="inference_experiments")
    args = parser.parse_args()

    # Setup
    checkpoint_path = Path(args.checkpoint)
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(exist_ok=True)

    print(f"Loading model from {checkpoint_path}...")

    # Load model (auto-detect Qwen2 vs Qwen3)
    if QWEN3_AVAILABLE:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(checkpoint_path),
            device_map={"": 0} if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    elif QWEN2_AVAILABLE:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(checkpoint_path),
            device_map={"": 0} if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        raise RuntimeError("No Qwen2/Qwen3 VL model available")

    processor = AutoProcessor.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    model.eval()

    # Load test data
    test_csv = REPO_ROOT / args.test_csv
    test_df = pd.read_csv(test_csv)
    image_dir = REPO_ROOT / args.image_dir

    print(f"Loaded {len(test_df)} test samples")

    # Run strategies
    if args.strategy == "all":
        # Test all strategies
        results = {}
        for name, strategy in STRATEGIES.items():
            predictions = run_inference(
                model, processor, test_df, image_dir, strategy, args.batch_size
            )

            # Save
            output_file = output_dir / f"submission_{name}.csv"
            submission = pd.DataFrame({
                "ID": test_df["ID"],
                "Target": predictions
            })
            submission.to_csv(output_file, index=False)
            print(f"  Saved: {output_file}")

            results[name] = predictions

        # Save summary
        summary = {
            "checkpoint": str(checkpoint_path),
            "strategies": list(STRATEGIES.keys()),
            "num_samples": len(test_df),
        }
        with open(output_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nGenerated {len(results)} submissions in {output_dir}/")
        print("Test each on leaderboard to find best strategy!")

    elif args.strategy == "ensemble":
        # Run top 3 strategies and ensemble
        top_strategies = ["greedy", "beam_3", "beam_5"]
        predictions_list = []

        for name in top_strategies:
            strategy = STRATEGIES[name]
            predictions = run_inference(
                model, processor, test_df, image_dir, strategy, args.batch_size
            )
            predictions_list.append(predictions)

        # Try different ensemble methods
        for method in ["voting", "shortest", "longest"]:
            ensembled = ensemble_predictions(predictions_list, method)

            output_file = output_dir / f"submission_ensemble_{method}.csv"
            submission = pd.DataFrame({
                "ID": test_df["ID"],
                "Target": ensembled
            })
            submission.to_csv(output_file, index=False)
            print(f"  Saved: {output_file}")

    else:
        # Single strategy
        strategy = STRATEGIES[args.strategy]
        predictions = run_inference(
            model, processor, test_df, image_dir, strategy, args.batch_size
        )

        output_file = output_dir / f"submission_{args.strategy}.csv"
        submission = pd.DataFrame({
            "ID": test_df["ID"],
            "Target": predictions
        })
        submission.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()

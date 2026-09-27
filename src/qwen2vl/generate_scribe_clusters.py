"""
VLM-based Scribe Clustering for HTR Split Strategy
Cluster historical documents by handwriting style to prevent train/val leakage.

Same scribe in both train/val = model memorizes their style instead of learning general OCR.
This script uses Qwen3-VL to analyze handwriting characteristics and cluster by scribe.

Output: scribe_clusters.csv with cluster IDs for stratified splitting
"""

import os
import sys
import json
import re
import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional, List

import yaml
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

# Handwriting analysis prompt - extracts scribe characteristics
HANDWRITING_ANALYSIS_PROMPT = """Analyze the handwriting style in this historical document image.

Rate these characteristics on a scale of 0-100:

1. **slant**: Backward slant (0), Vertical (50), Forward slant (100)
2. **spacing**: Very tight spacing (0), Normal (50), Very wide spacing (100)
3. **line_quality**: Shaky/irregular (0), Normal (50), Very smooth/precise (100)
4. **letter_size**: Small letters (0), Medium (50), Large letters (100)
5. **formality**: Casual/rushed (0), Normal (50), Very formal/careful (100)
6. **stroke_weight**: Thin/light strokes (0), Medium (50), Thick/heavy strokes (100)
7. **consistency**: Highly variable (0), Normal (50), Very consistent (100)
8. **flourish**: Minimal decoration (0), Some flourishes (50), Highly decorated (100)

Respond ONLY with a JSON object in this exact format:
{
    "slant": <0-100>,
    "spacing": <0-100>,
    "line_quality": <0-100>,
    "letter_size": <0-100>,
    "formality": <0-100>,
    "stroke_weight": <0-100>,
    "consistency": <0-100>,
    "flourish": <0-100>
}"""


# ─────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────

def load_model_and_processor(model_name: str, max_pixels: int = 2016000):
    """Load Qwen3-VL model and processor for handwriting analysis."""
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
# HANDWRITING ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyze_handwriting_batch(
    image_paths: list[Path],
    model,
    processor,
    max_pixels: int = 2016000,
    max_new_tokens: int = 256,
    verbose: bool = False,
) -> list[Optional[Dict[str, float]]]:
    """
    Analyze handwriting characteristics using VLM (batched).

    Returns:
        List of feature dicts (one per image), None for failed analyses
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
                        {"type": "text", "text": HANDWRITING_ANALYSIS_PROMPT},
                    ],
                }
            ]
            for img in images
        ]

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        # Set left-padding for batched generation
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

            # Generate analyses
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

            features = parse_handwriting_response(response)

            if features is None and verbose:
                print(f"  ⚠️  Failed to parse JSON from response for {image_paths[original_idx].name}")

            results[original_idx] = features

        return results

    except Exception as e:
        print(f"  ⚠️  Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        return [None] * len(image_paths)


def parse_handwriting_response(response: str) -> Optional[Dict[str, float]]:
    """
    Parse VLM response to extract handwriting features.

    Returns dict with: slant, spacing, line_quality, letter_size, formality,
                       stroke_weight, consistency, flourish
    """
    # Try to find JSON object
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

    if not json_match:
        return None

    json_str = json_match.group(0)

    # Clean up common formatting issues
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*', '', json_str)

    try:
        data = json.loads(json_str)

        # Expected fields
        required_fields = [
            "slant", "spacing", "line_quality", "letter_size",
            "formality", "stroke_weight", "consistency", "flourish"
        ]

        features = {}

        for field in required_fields:
            if field in data:
                try:
                    value = float(data[field])
                    features[field] = max(0.0, min(100.0, value))
                except (ValueError, TypeError):
                    return None
            else:
                # Missing field
                return None

        return features

    except (json.JSONDecodeError, ValueError, KeyError):
        return None


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
    batch_size: int = 4,
    save_interval: int = 50,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Process entire dataset and extract handwriting features.
    """
    # Load existing results if resuming
    existing_results = {}
    if resume and output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)

            # Validate columns
            required_cols = {"ID", "slant", "spacing", "line_quality", "letter_size",
                           "formality", "stroke_weight", "consistency", "flourish", "success"}
            if not required_cols.issubset(existing_df.columns):
                missing = required_cols - set(existing_df.columns)
                print(f"⚠️  Existing CSV missing columns {missing} - starting fresh")
            else:
                # Only load successful analyses
                valid_results = existing_df[existing_df["success"] == True].copy()
                existing_results = valid_results.set_index("ID").to_dict("index")

                n_total = len(existing_df)
                n_valid = len(valid_results)
                n_failed = n_total - n_valid

                print(f"Resuming: {n_valid} valid results loaded ({n_failed} failed analyses will be retried)")
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
    for batch_start in tqdm(range(0, len(unprocessed), batch_size), desc="Analyzing handwriting (batched)"):
        batch_items = unprocessed[batch_start:batch_start + batch_size]
        batch_ids = [img_id for _, img_id in batch_items]

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

        # Analyze handwriting for batch
        batch_features = analyze_handwriting_batch(
            batch_paths, model, processor, max_pixels=max_pixels, verbose=verbose
        )

        # Process results
        for (idx, img_id), features in zip(valid_batch_items, batch_features):
            analysis_succeeded = features is not None

            if features is None:
                print(f"  ⚠️  Analysis failed: {img_id}")
                failed_ids.append(img_id)
                # Add placeholder features (neutral)
                features = {
                    "slant": 50.0,
                    "spacing": 50.0,
                    "line_quality": 50.0,
                    "letter_size": 50.0,
                    "formality": 50.0,
                    "stroke_weight": 50.0,
                    "consistency": 50.0,
                    "flourish": 50.0,
                }

            # Store result
            result = {
                "ID": img_id,
                "slant": features["slant"],
                "spacing": features["spacing"],
                "line_quality": features["line_quality"],
                "letter_size": features["letter_size"],
                "formality": features["formality"],
                "stroke_weight": features["stroke_weight"],
                "consistency": features["consistency"],
                "flourish": features["flourish"],
                "success": analysis_succeeded,
            }
            results.append(result)

        # Save checkpoint
        if len(results) % save_interval < batch_size:
            df = pd.DataFrame(results)
            if len(df) > 0 and "success" in df.columns:
                backup_path = output_csv.with_suffix('.csv.bak')
                if output_csv.exists():
                    output_csv.rename(backup_path)
                df.to_csv(output_csv, index=False)
                if output_csv.exists() and backup_path.exists():
                    backup_path.unlink()
                print(f"  Checkpoint saved: {len(results)} images processed ({df['success'].sum()} successful)")

    # Final save
    df = pd.DataFrame(results)
    backup_path = output_csv.with_suffix('.csv.bak')
    if output_csv.exists():
        output_csv.rename(backup_path)
    df.to_csv(output_csv, index=False)
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
# CLUSTERING
# ─────────────────────────────────────────────────────────────

def cluster_scribes(df: pd.DataFrame, method: str = "hierarchical", n_clusters: int = None) -> pd.DataFrame:
    """
    Cluster documents by handwriting style.

    Args:
        df: DataFrame with handwriting features
        method: "hierarchical", "kmeans", or "dbscan"
        n_clusters: Number of clusters (for hierarchical/kmeans)

    Returns:
        DataFrame with scribe_cluster column
    """
    # Filter to successful analyses
    df_valid = df[df["success"] == True].copy()

    if len(df_valid) == 0:
        print("⚠️  No successful analyses to cluster")
        return df

    # Feature columns
    feature_cols = ["slant", "spacing", "line_quality", "letter_size",
                    "formality", "stroke_weight", "consistency", "flourish"]

    X = df_valid[feature_cols].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Auto-determine n_clusters if not specified
    if n_clusters is None:
        # Rule of thumb: sqrt(n_samples)
        n_clusters = max(5, min(50, int(np.sqrt(len(df_valid)))))
        print(f"Auto-selected n_clusters: {n_clusters} (from {len(df_valid)} samples)")

    # Cluster
    print(f"Clustering with method: {method}, n_clusters: {n_clusters}")

    if method == "hierarchical":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = clusterer.fit_predict(X_scaled)

    elif method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)

    elif method == "dbscan":
        # DBSCAN auto-determines clusters
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"  DBSCAN found {n_clusters} clusters ({n_noise} noise points)")

    else:
        raise ValueError(f"Unknown method: {method}")

    # Assign cluster labels
    df_valid["scribe_cluster"] = labels

    # Merge back to original dataframe
    df = df.merge(df_valid[["ID", "scribe_cluster"]], on="ID", how="left")

    # Assign failed/missing to cluster -1
    df["scribe_cluster"] = df["scribe_cluster"].fillna(-1).astype(int)

    return df


def print_cluster_stats(df: pd.DataFrame):
    """Print clustering statistics."""
    print(f"\n{'='*70}")
    print("SCRIBE CLUSTERING STATISTICS")
    print(f"{'='*70}")

    cluster_counts = df["scribe_cluster"].value_counts().sort_index()

    print(f"\nTotal clusters: {len(cluster_counts)}")
    print(f"Total documents: {len(df)}")

    print(f"\nCluster sizes:")
    for cluster_id, count in cluster_counts.items():
        if cluster_id == -1:
            print(f"  Cluster {cluster_id:3d} (failed/noise): {count:4d} documents")
        else:
            print(f"  Cluster {cluster_id:3d}: {count:4d} documents")

    # Statistics
    valid_clusters = cluster_counts[cluster_counts.index != -1]
    if len(valid_clusters) > 0:
        print(f"\nCluster size statistics (excluding failed):")
        print(f"  Mean:   {valid_clusters.mean():.1f}")
        print(f"  Median: {valid_clusters.median():.0f}")
        print(f"  Min:    {valid_clusters.min()}")
        print(f"  Max:    {valid_clusters.max()}")

    print(f"{'='*70}\n")


def visualize_clusters(df: pd.DataFrame, output_dir: Path, features_csv: Path):
    """Create visualization of scribe clusters."""
    df_valid = df[df["success"] == True].copy()

    if len(df_valid) == 0:
        print("⚠️  No successful analyses to visualize")
        return

    feature_cols = ["slant", "spacing", "line_quality", "letter_size",
                    "formality", "stroke_weight", "consistency", "flourish"]

    X = df_valid[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_valid["pca1"] = X_pca[:, 0]
    df_valid["pca2"] = X_pca[:, 1]

    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df_valid["pca1"],
        df_valid["pca2"],
        c=df_valid["scribe_cluster"],
        cmap='tab20',
        alpha=0.6,
        s=50
    )
    plt.colorbar(scatter, label="Scribe Cluster")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("Scribe Clusters (PCA Visualization)")
    plt.grid(alpha=0.3)

    plot_path = output_dir / "scribe_clusters_pca.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {plot_path}")
    plt.close()

    # Feature distributions per cluster
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        df_valid.boxplot(column=feature, by="scribe_cluster", ax=ax)
        ax.set_title(feature.replace('_', ' ').title())
        ax.set_xlabel("Cluster")
        plt.sca(ax)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plot_path = output_dir / "scribe_features_by_cluster.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature distributions: {plot_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate scribe clusters using Qwen3-VL handwriting analysis"
    )
    parser.add_argument(
        "--config",
        default="config_qwen3_8b.yaml",
        help="Config file (in configs/ directory)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Qwen3-VL model to use"
    )
    parser.add_argument(
        "--features-output",
        default="dataset/handwriting_features.csv",
        help="Output CSV for handwriting features"
    )
    parser.add_argument(
        "--clusters-output",
        default="dataset/scribe_clusters.csv",
        help="Output CSV for scribe clusters"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Process only first N images (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print VLM responses"
    )
    parser.add_argument(
        "--cluster-method",
        choices=["hierarchical", "kmeans", "dbscan"],
        default="hierarchical",
        help="Clustering method"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (auto if not specified)"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip VLM analysis, just cluster existing features"
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / "configs" / args.config
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    image_dir = REPO_ROOT / cfg["data"]["image_dir"]
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)

    features_csv = REPO_ROOT / args.features_output
    clusters_csv = REPO_ROOT / args.clusters_output

    # Get image IDs from Train.csv
    train_csv = REPO_ROOT / cfg["data"]["train_csv"]
    if not train_csv.exists():
        print(f"Error: Train CSV not found: {train_csv}")
        sys.exit(1)

    train_df = pd.read_csv(train_csv)
    image_ids = sorted(train_df["ID"].astype(str).tolist())

    if args.subset:
        image_ids = image_ids[:args.subset]
        print(f"Processing subset: {args.subset} images")

    print(f"Found {len(image_ids)} images in {train_csv.name}")

    # Step 1: Extract handwriting features (or load existing)
    if not args.skip_analysis:
        max_pixels = cfg["training"].get("max_pixels", 2016000)
        model, processor = load_model_and_processor(args.model, max_pixels=max_pixels)

        print(f"\nExtracting handwriting features...")
        print(f"Output: {features_csv}")
        print(f"Batch size: {args.batch_size}")
        print(f"Resume: {not args.no_resume}\n")

        df_features = process_dataset(
            image_dir=image_dir,
            image_ids=image_ids,
            model=model,
            processor=processor,
            output_csv=features_csv,
            resume=not args.no_resume,
            max_pixels=max_pixels,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
    else:
        print(f"Loading existing features from {features_csv}")
        if not features_csv.exists():
            print(f"Error: Features file not found: {features_csv}")
            sys.exit(1)
        df_features = pd.read_csv(features_csv)

    # Step 2: Cluster by handwriting style
    print(f"\nClustering scribes...")
    df_clustered = cluster_scribes(
        df_features,
        method=args.cluster_method,
        n_clusters=args.n_clusters
    )

    # Save clusters
    df_clustered.to_csv(clusters_csv, index=False)
    print(f"✓ Saved scribe clusters: {clusters_csv}")

    # Step 3: Print statistics
    print_cluster_stats(df_clustered)

    # Step 4: Visualize
    output_dir = clusters_csv.parent
    visualize_clusters(df_clustered, output_dir, features_csv)

    print(f"\n✓ Scribe clustering complete!")
    print(f"  Features: {features_csv}")
    print(f"  Clusters: {clusters_csv}")
    print(f"\nTo use in training, set in config:")
    print(f"  data:")
    print(f"    cluster_csv: \"{args.clusters_output}\"")
    print(f"    group_col: \"scribe_cluster\"\n")


if __name__ == "__main__":
    main()

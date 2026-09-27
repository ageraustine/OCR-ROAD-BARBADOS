"""
Find optimal number of scribe clusters using elbow method and silhouette score.
Run this after generating handwriting_features.csv to determine best n_clusters.

Usage:
    python optimize_cluster_count.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys

sys.path.append(str(Path(__file__).parent))
from data_utils import REPO_ROOT


def find_optimal_clusters(features_csv: Path, max_clusters: int = 50):
    """Find optimal number of clusters using elbow + silhouette analysis."""

    df = pd.read_csv(features_csv)
    df_valid = df[df["success"] == True]

    print(f"Loaded {len(df_valid)} valid handwriting analyses")

    feature_cols = ["slant", "spacing", "line_quality", "letter_size",
                    "formality", "stroke_weight", "consistency", "flourish"]

    X = df_valid[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test range of cluster counts
    min_clusters = max(2, int(np.sqrt(len(df_valid)) * 0.5))
    max_clusters = min(max_clusters, len(df_valid) // 10)
    cluster_range = range(min_clusters, max_clusters + 1, max(1, (max_clusters - min_clusters) // 20))

    print(f"Testing cluster counts: {min(cluster_range)} to {max(cluster_range)}")

    inertias = []
    silhouettes = []

    for n in cluster_range:
        # K-means for inertia
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)

        # Silhouette score
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)

        print(f"  n={n:3d}: inertia={kmeans.inertia_:8.1f}, silhouette={sil:.3f}")

    # Find elbow using second derivative
    inertias_arr = np.array(inertias)
    diffs = np.diff(inertias_arr)
    second_diffs = np.diff(diffs)
    elbow_idx = np.argmax(second_diffs) + 2  # +2 because of double diff
    elbow_n = list(cluster_range)[elbow_idx]

    # Best silhouette
    best_sil_idx = np.argmax(silhouettes)
    best_sil_n = list(cluster_range)[best_sil_idx]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    ax1.plot(cluster_range, inertias, 'bo-', linewidth=2)
    ax1.axvline(elbow_n, color='r', linestyle='--', label=f'Elbow at n={elbow_n}')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
    ax1.set_title('Elbow Method')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Silhouette plot
    ax2.plot(cluster_range, silhouettes, 'go-', linewidth=2)
    ax2.axvline(best_sil_n, color='r', linestyle='--', label=f'Best at n={best_sil_n}')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    output_path = features_csv.parent / "cluster_optimization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_path}")

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    print(f"Elbow method suggests:     n = {elbow_n}")
    print(f"Best silhouette score:     n = {best_sil_n} (score: {max(silhouettes):.3f})")
    print(f"Auto (sqrt rule):          n = {int(np.sqrt(len(df_valid)))}")
    print(f"\nRecommended: n = {elbow_n} (elbow method is most reliable)")
    print(f"{'='*60}\n")

    return elbow_n, best_sil_n


if __name__ == "__main__":
    features_csv = REPO_ROOT / "dataset" / "handwriting_features.csv"

    if not features_csv.exists():
        print(f"Error: {features_csv} not found")
        print("Run: python generate_scribe_clusters.py first")
        sys.exit(1)

    find_optimal_clusters(features_csv)

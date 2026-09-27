"""
Feature Selection for Optimal Train/Val Split Strategy

Analyzes which features are most predictive of OCR difficulty and should be
used for stratification. Uses correlation with CER, mutual information, and
feature independence analysis.

Critical for competition: proper stratification = reliable validation = better hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
import sys

sys.path.append(str(Path(__file__).parent))
from data_utils import REPO_ROOT, compute_stratification_bins, load_and_prepare_dataframe
import yaml


def extract_all_features(df: pd.DataFrame, quality_csv: Path) -> pd.DataFrame:
    """Extract all potential stratification features."""

    # Merge with quality scores
    if quality_csv.exists():
        quality_df = pd.read_csv(quality_csv)
        quality_df = quality_df[quality_df["success"] == True]
        df = df.merge(quality_df, on="ID", how="left")
        print(f"✓ Merged quality scores: {len(quality_df)} docs")
    else:
        print(f"⚠️  Quality scores not found: {quality_csv}")

    # Text-based features (already computed by compute_stratification_bins)
    df_annotated = compute_stratification_bins(df, {"fuzzy_duplicate_threshold": 0.90})

    # Image dimension features (if available)
    # Note: Would need to read images for this - skip for now unless requested

    # Composite features
    if "physical_damage" in df.columns:
        df["degradation_severity"] = (
            df["physical_damage"] * 0.3 +
            df["ink_degradation"] * 0.4 +
            df["paper_condition"] * 0.3
        )

    if "_text_len" in df.columns:
        # Length categories
        df["length_category"] = pd.cut(
            df["_text_len"],
            bins=[0, 50, 150, 300, 10000],
            labels=["very_short", "short", "medium", "long"]
        )

    return df


def analyze_feature_predictiveness(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    """
    Analyze which features are most predictive of OCR difficulty.

    If target_col (like 'cer' or 'wer') is available, use it.
    Otherwise, use proxy measures (text complexity, quality scores).
    """

    # Define candidate features
    numeric_features = []
    categorical_features = []

    # Quality scores
    quality_features = ["composite_score", "physical_damage", "ink_degradation",
                       "paper_condition", "text_readability", "degradation_severity"]

    # Text complexity
    text_features = ["_text_len", "_digit_density", "_uppercase_ratio",
                    "_lexical_diversity", "_special_char_density", "_avg_word_length"]

    # Categorical
    cat_features = ["_text_type", "_has_rare", "_has_digit", "_has_upper",
                   "_has_interlineation", "_has_superscript", "length_category"]

    # Check which features exist
    for feat in quality_features + text_features:
        if feat in df.columns:
            numeric_features.append(feat)

    for feat in cat_features:
        if feat in df.columns:
            categorical_features.append(feat)

    print(f"\nAnalyzing {len(numeric_features)} numeric + {len(categorical_features)} categorical features")

    # Create proxy target if real target not available
    if target_col is None or target_col not in df.columns:
        print("\n⚠️  No ground-truth CER/WER available - using proxy difficulty score")
        # Proxy: composite difficulty from quality + text complexity
        proxy = 0
        if "composite_score" in df.columns:
            proxy += df["composite_score"].fillna(df["composite_score"].median())
        if "_text_len" in df.columns:
            # Longer text = harder (slightly)
            proxy += (df["_text_len"] / 1000) * 5
        if "_digit_density" in df.columns:
            # Digits harder to OCR
            proxy += df["_digit_density"] * 20
        if "_special_char_density" in df.columns:
            # Special chars harder
            proxy += df["_special_char_density"] * 15

        df["_proxy_difficulty"] = proxy
        target_col = "_proxy_difficulty"
        print(f"  Using proxy difficulty: mean={proxy.mean():.2f}, std={proxy.std():.2f}")

    # Filter to valid samples
    df_valid = df.dropna(subset=[target_col] + numeric_features[:5])  # At least some features

    if len(df_valid) < 100:
        print(f"⚠️  Too few valid samples: {len(df_valid)}")
        return pd.DataFrame()

    print(f"  Valid samples: {len(df_valid)}/{len(df)}")

    # Analysis 1: Correlation with target
    correlations = {}
    for feat in numeric_features:
        if feat in df_valid.columns:
            valid_data = df_valid[[feat, target_col]].dropna()
            if len(valid_data) > 10:
                corr, pval = spearmanr(valid_data[feat], valid_data[target_col])
                correlations[feat] = {"correlation": abs(corr), "pval": pval}

    # Analysis 2: Mutual Information (captures non-linear relationships)
    mi_scores = {}
    X = df_valid[numeric_features].fillna(df_valid[numeric_features].median())
    y = df_valid[target_col]

    if len(X) > 50:
        mi = mutual_info_regression(X, y, random_state=42)
        for feat, score in zip(numeric_features, mi):
            mi_scores[feat] = score

    # Analysis 3: Random Forest feature importance (captures interactions)
    rf_importance = {}
    if len(df_valid) > 100:
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        for feat, importance in zip(numeric_features, rf.feature_importances_):
            rf_importance[feat] = importance

    # Combine results
    results = []
    for feat in numeric_features:
        row = {"feature": feat}
        if feat in correlations:
            row["correlation"] = correlations[feat]["correlation"]
            row["pval"] = correlations[feat]["pval"]
        if feat in mi_scores:
            row["mutual_info"] = mi_scores[feat]
        if feat in rf_importance:
            row["rf_importance"] = rf_importance[feat]
        results.append(row)

    results_df = pd.DataFrame(results)

    # Normalize scores to [0, 1] and compute composite
    for col in ["correlation", "mutual_info", "rf_importance"]:
        if col in results_df.columns:
            max_val = results_df[col].max()
            if max_val > 0:
                results_df[f"{col}_norm"] = results_df[col] / max_val

    # Composite score (equal weight)
    score_cols = [c for c in results_df.columns if c.endswith("_norm")]
    if score_cols:
        results_df["composite_score"] = results_df[score_cols].mean(axis=1)
        results_df = results_df.sort_values("composite_score", ascending=False)

    return results_df


def analyze_feature_independence(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Analyze correlation between features to identify redundancy.

    Highly correlated features (>0.7) are redundant - only keep one.
    """
    df_valid = df[features].dropna()

    if len(df_valid) < 100:
        print(f"⚠️  Too few samples for independence analysis: {len(df_valid)}")
        return pd.DataFrame()

    # Compute correlation matrix
    corr_matrix = df_valid.corr(method='spearman')

    return corr_matrix


def plot_feature_analysis(predictiveness_df: pd.DataFrame, independence_matrix: pd.DataFrame, output_dir: Path):
    """Create visualizations of feature analysis."""

    # Plot 1: Feature predictiveness
    if not predictiveness_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        plot_df = predictiveness_df.head(15)  # Top 15 features

        x = np.arange(len(plot_df))
        width = 0.25

        if "correlation_norm" in plot_df.columns:
            ax.bar(x - width, plot_df["correlation_norm"], width, label="Correlation", alpha=0.8)
        if "mutual_info_norm" in plot_df.columns:
            ax.bar(x, plot_df["mutual_info_norm"], width, label="Mutual Info", alpha=0.8)
        if "rf_importance_norm" in plot_df.columns:
            ax.bar(x + width, plot_df["rf_importance_norm"], width, label="RF Importance", alpha=0.8)

        ax.set_xlabel("Feature")
        ax.set_ylabel("Normalized Score")
        ax.set_title("Feature Predictiveness Analysis\n(Higher = More Predictive of OCR Difficulty)")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["feature"], rotation=45, ha="right")
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = output_dir / "feature_predictiveness.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()

    # Plot 2: Feature correlation matrix
    if not independence_matrix.empty and HAS_SEABORN:
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            independence_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"label": "Spearman Correlation"}
        )

        ax.set_title("Feature Independence Analysis\n(Highly correlated features are redundant)")
        plt.tight_layout()
        plot_path = output_dir / "feature_correlation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
    elif not independence_matrix.empty:
        print("  ℹ️  Skipping correlation heatmap (seaborn not available)")


def recommend_features(predictiveness_df: pd.DataFrame, independence_matrix: pd.DataFrame, n_features: int = 3) -> list:
    """
    Recommend optimal features for stratification.

    Strategy:
    1. Rank by predictiveness
    2. Remove redundant features (correlation > 0.7)
    3. Select top N independent + predictive features
    """

    if predictiveness_df.empty:
        return []

    # Sort by composite score
    ranked = predictiveness_df.sort_values("composite_score", ascending=False)

    selected = []
    for _, row in ranked.iterrows():
        feat = row["feature"]

        # Check if redundant with already selected features
        is_redundant = False
        if not independence_matrix.empty:
            for selected_feat in selected:
                if feat in independence_matrix.columns and selected_feat in independence_matrix.index:
                    corr = abs(independence_matrix.loc[selected_feat, feat])
                    if corr > 0.7:  # Redundancy threshold
                        is_redundant = True
                        break

        if not is_redundant:
            selected.append(feat)

        if len(selected) >= n_features:
            break

    return selected


def main():
    # Load config
    config_path = REPO_ROOT / "src/qwen2vl/configs/config_qwen3_8b.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load training data
    print("Loading training data...")
    df = load_and_prepare_dataframe(cfg["data"])

    # Load quality scores
    quality_csv = REPO_ROOT / "dataset/document_condition_v2.csv"

    # Extract features
    print("\nExtracting features...")
    df = extract_all_features(df, quality_csv)

    # Analyze predictiveness
    print("\n" + "="*70)
    print("FEATURE PREDICTIVENESS ANALYSIS")
    print("="*70)
    predictiveness_df = analyze_feature_predictiveness(df)

    if not predictiveness_df.empty:
        print("\nTop 10 Most Predictive Features:")
        print(predictiveness_df.head(10).to_string(index=False))

    # Analyze independence
    print("\n" + "="*70)
    print("FEATURE INDEPENDENCE ANALYSIS")
    print("="*70)

    top_features = predictiveness_df.head(10)["feature"].tolist() if not predictiveness_df.empty else []
    independence_matrix = analyze_feature_independence(df, top_features)

    if not independence_matrix.empty:
        print("\nHighly Correlated Feature Pairs (|r| > 0.7):")
        for i in range(len(independence_matrix)):
            for j in range(i+1, len(independence_matrix)):
                corr = independence_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    feat1 = independence_matrix.index[i]
                    feat2 = independence_matrix.columns[j]
                    print(f"  {feat1:30s} <-> {feat2:30s}: {corr:+.3f}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    for n in [2, 3, 4]:
        recommended = recommend_features(predictiveness_df, independence_matrix, n_features=n)
        print(f"\nTop {n} features for stratification:")
        for i, feat in enumerate(recommended, 1):
            score = predictiveness_df[predictiveness_df["feature"] == feat]["composite_score"].values[0]
            print(f"  {i}. {feat:30s} (score: {score:.3f})")

    # Current stratification
    print("\n" + "="*70)
    print("CURRENT STRATIFICATION")
    print("="*70)
    print("Currently using:")
    print("  1. condition_score (from composite_score)")
    print("  2. text_type (4 categories)")
    print("  3. rare_vocab (binary)")
    print(f"\nTotal bins: 3 × 4 × 2 = 24")

    # Visualizations
    output_dir = REPO_ROOT / "dataset"
    plot_feature_analysis(predictiveness_df, independence_matrix, output_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Plots saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review recommended features vs current stratification")
    print("2. Test alternative stratification strategies")
    print("3. Measure val CER correlation with each strategy")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

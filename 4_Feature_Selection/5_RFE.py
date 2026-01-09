import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_curve, auc, precision_score, f1_score,
                             matthews_corrcoef, accuracy_score, recall_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import warnings
import gc
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import json
import joblib
from scipy import stats

warnings.filterwarnings('ignore')

# Set font family globally
plt.rcParams['font.family'] = 'Times New Roman'
dpi = 1000
plt.rcParams['figure.dpi'] = dpi

# Base directory
BASE_DIR = "AttBiLSTM_Analysis_With_RFE_FS"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'plots'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'feature_selection'), exist_ok=True)

# Device setup
device = torch.device("cpu")
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device detected, using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device detected, using CUDA")
    print(f"Using device: {device}")
except:
    print("Error detecting device capabilities, defaulting to CPU")

# Load and prepare data
print("Loading and preparing data...")
try:
    data = pd.read_csv('../dataset/Combined_Common_Genes_With_Target_ML.csv')
    print(f"Dataset info:")
    print(f"Shape: {data.shape}")
    print(f"Target column: {data.columns[-1]}")

    # Separate features and target
    X = data.iloc[:, :-1].values  # All columns except last
    y = data.iloc[:, -1].values  # Last column (target)
    feature_names = data.columns[:-1].tolist()  # Store feature names

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of original features: {len(feature_names)}")

    # Check target distribution
    unique_targets, counts = np.unique(y, return_counts=True)
    print(f"Target distribution:")
    for target, count in zip(unique_targets, counts):
        print(f"  Class {target}: {count} samples ({count / len(y) * 100:.2f}%)")

    num_classes = len(unique_targets)

except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Data preprocessing and scaling
print("Preprocessing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initial split for feature selection
X_train_initial, X_temp, y_train_initial, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
X_val_initial, X_test_initial, y_val_initial, y_test_initial = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Initial training set: {X_train_initial.shape[0]} samples")
print(f"Initial validation set: {X_val_initial.shape[0]} samples")
print(f"Initial test set: {X_test_initial.shape[0]} samples")


def perform_rfe_feature_selection(X_train, y_train, X_val, y_val,
                                  feature_names, n_features=2000,
                                  estimator_type='logistic', step=1):
    """
    Perform Recursive Feature Elimination (RFE) based feature selection

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: List of feature names
        n_features: Number of features to select
        estimator_type: Type of estimator ('logistic', 'rf', 'svm')
        step: Number of features to remove at each iteration

    Returns:
        selected_indices: Indices of selected features
        selected_features: Names of selected features
        feature_importance: Feature importance scores
    """
    print("=" * 60)
    print("PERFORMING RECURSIVE FEATURE ELIMINATION (RFE) FEATURE SELECTION")
    print("=" * 60)

    print(f"Original number of FEATURES (columns): {X_train.shape[1]}")
    print(f"Number of samples (rows): {X_train.shape[0]}")
    print(f"Target number of FEATURES to select: {n_features}")
    print(f"Estimator type: {estimator_type}")
    print(f"Step size: {step}")

    # If we already have fewer features than target, return all
    if X_train.shape[1] <= n_features:
        print(f"Dataset already has {X_train.shape[1]} features, which is <= {n_features}")
        selected_indices = np.arange(X_train.shape[1])
        selected_features = feature_names
        feature_importance = np.ones(X_train.shape[1])  # Dummy values
        return selected_indices, selected_features, feature_importance

    # Step 1: Choose the base estimator
    print("Setting up base estimator...")

    if estimator_type == 'logistic':
        estimator = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear',
            class_weight='balanced'
        )
        print("Using Logistic Regression as base estimator")

    elif estimator_type == 'rf':
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        print("Using Random Forest as base estimator")

    elif estimator_type == 'svm':
        estimator = SVC(
            kernel='linear',
            random_state=42,
            class_weight='balanced'
        )
        print("Using Linear SVM as base estimator")

    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")

    # Step 2: Perform RFE
    print("Performing Recursive Feature Elimination...")
    print("This may take several minutes for large datasets...")

    start_time = time.time()

    # Create RFE selector
    rfe_selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
        step=step,
        verbose=1
    )

    # Fit RFE
    print("Fitting RFE selector...")
    X_train_selected = rfe_selector.fit_transform(X_train, y_train)

    rfe_calculation_time = time.time() - start_time
    print(f"RFE calculation completed in {rfe_calculation_time:.2f} seconds")

    # Get the selected feature indices
    selected_indices = np.where(rfe_selector.support_)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    print(f"Selected {len(selected_indices)} features (columns)")
    print(f"Selected feature indices range: {selected_indices.min()} to {selected_indices.max()}")
    print(f"Top 5 selected features: {selected_features[:5]}")

    # Get feature rankings and importance scores
    feature_rankings = rfe_selector.ranking_

    # For feature importance, we'll use the ranking (inverted so higher is better)
    max_rank = np.max(feature_rankings)
    feature_importance = (max_rank + 1 - feature_rankings) / max_rank

    # If the estimator has feature_importances_ or coef_, use those for selected features
    if hasattr(rfe_selector.estimator_, 'feature_importances_'):
        # For tree-based models
        estimator_importance = rfe_selector.estimator_.feature_importances_
        print("Using tree-based feature importances")
    elif hasattr(rfe_selector.estimator_, 'coef_'):
        # For linear models
        estimator_importance = np.abs(rfe_selector.estimator_.coef_[0])
        print("Using linear model coefficients")
    else:
        # Fallback to ranking-based importance
        estimator_importance = feature_importance[selected_indices]
        print("Using ranking-based importance")

    # Create final importance scores
    final_importance = np.zeros(len(feature_names))
    final_importance[selected_indices] = estimator_importance

    # Print some statistics
    selected_importance = final_importance[selected_indices]

    print(f"Feature importance statistics for selected features:")
    print(f"  Mean importance: {np.mean(selected_importance):.6f}")
    print(f"  Std importance: {np.std(selected_importance):.6f}")
    print(f"  Min importance: {np.min(selected_importance):.6f}")
    print(f"  Max importance: {np.max(selected_importance):.6f}")
    print(f"  Median importance: {np.median(selected_importance):.6f}")

    # Analyze ranking distribution
    selected_rankings = feature_rankings[selected_indices]
    print(f"Feature ranking statistics for selected features:")
    print(f"  All selected features have ranking: 1 (highest priority)")
    print(f"  Non-selected features have rankings: 2 to {max_rank}")

    # Step 3: Evaluate the selection
    # Train the estimator on selected features for evaluation
    X_val_selected = rfe_selector.transform(X_val)

    eval_model = LogisticRegression(random_state=42, max_iter=1000)
    eval_model.fit(X_train_selected, y_train)

    train_score = eval_model.score(X_train_selected, y_train)
    val_score = eval_model.score(X_val_selected, y_val)

    print(f"Logistic Regression on RFE selected features:")
    print(f"  Train Accuracy: {train_score:.4f}")
    print(f"  Validation Accuracy: {val_score:.4f}")

    # Step 4: Save RFE analysis results
    rfe_results = {
        'selected_indices': selected_indices.tolist(),
        'selected_features': selected_features,
        'feature_rankings': feature_rankings.tolist(),
        'feature_importance': final_importance.tolist(),
        'selected_importance': selected_importance.tolist(),
        'train_score': float(train_score),
        'val_score': float(val_score),
        'original_features': len(feature_names),
        'selected_features_count': len(selected_indices),
        'rfe_calculation_time': float(rfe_calculation_time),
        'estimator_type': estimator_type,
        'step_size': step,
        'mean_importance': float(np.mean(selected_importance)),
        'std_importance': float(np.std(selected_importance)),
        'min_importance': float(np.min(selected_importance)),
        'max_importance': float(np.max(selected_importance)),
        'median_importance': float(np.median(selected_importance))
    }

    # Save RFE results
    rfe_results_path = os.path.join(BASE_DIR, 'feature_selection', 'rfe_results.json')
    with open(rfe_results_path, 'w') as f:
        json.dump(rfe_results, f, indent=2)

    # Save the RFE selector
    rfe_selector_path = os.path.join(BASE_DIR, 'feature_selection', 'rfe_selector.joblib')
    joblib.dump(rfe_selector, rfe_selector_path)

    # Create feature importance plot
    plot_rfe_feature_importance(final_importance, feature_names, selected_indices,
                                feature_rankings, n_features)

    print(f"RFE analysis results saved to: {rfe_results_path}")
    print(f"RFE selector saved to: {rfe_selector_path}")

    return selected_indices, selected_features, final_importance


def plot_rfe_feature_importance(feature_importance, feature_names, selected_indices,
                                feature_rankings, n_features):
    """Plot RFE feature importance and rankings"""

    # Plot top features by importance
    top_indices = selected_indices[:min(50, n_features)]  # Show top 50 features
    top_importance = feature_importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    top_rankings = feature_rankings[top_indices]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # ================== SUBPLOT 1: Feature Importance ==================
    y_pos = np.arange(len(top_names))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_names)))

    bars = ax1.barh(y_pos, top_importance, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_names, fontsize=8)
    ax1.set_xlabel('Feature Importance Score', fontsize=14)
    ax1.set_ylabel('Features', fontsize=14)
    ax1.set_title(f'Top {len(top_names)} Features by RFE Importance', fontsize=16, pad=20)
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', ha='left', va='center', fontsize=8)

    # ================== SUBPLOT 2: Ranking Distribution ==================
    ax2.hist(feature_rankings, bins=50, color='skyblue', edgecolor='navy', alpha=0.7)
    ax2.axvline(1, color='red', linestyle='--', linewidth=2,
                label=f'Selected Features (Rank=1, n={len(selected_indices)})')
    ax2.set_xlabel('RFE Ranking', fontsize=14)
    ax2.set_ylabel('Number of Features', fontsize=14)
    ax2.set_title('Distribution of RFE Rankings', fontsize=16, pad=20)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ================== SUBPLOT 3: Importance vs Ranking ==================
    # Scatter plot of importance vs ranking for all features
    scatter_colors = ['red' if i in selected_indices else 'blue'
                      for i in range(len(feature_importance))]
    ax3.scatter(feature_rankings, feature_importance, c=scatter_colors, alpha=0.6)
    ax3.set_xlabel('RFE Ranking', fontsize=14)
    ax3.set_ylabel('Feature Importance', fontsize=14)
    ax3.set_title('Feature Importance vs RFE Ranking', fontsize=16, pad=20)
    ax3.grid(alpha=0.3)

    # Add legend
    ax3.scatter([], [], c='red', label='Selected Features')
    ax3.scatter([], [], c='blue', label='Non-selected Features')
    ax3.legend()

    # ================== SUBPLOT 4: Cumulative Feature Selection ==================
    # Show how many features were selected at each ranking level
    unique_ranks = np.unique(feature_rankings)
    cumulative_selected = []
    for rank in unique_ranks:
        cumulative_selected.append(np.sum(feature_rankings <= rank))

    ax4.plot(unique_ranks, cumulative_selected, 'b-', linewidth=3, marker='o')
    ax4.axhline(n_features, color='red', linestyle='--', linewidth=2,
                label=f'Target Features ({n_features})')
    ax4.axvline(1, color='green', linestyle='--', linewidth=2,
                label='Selected Features (Rank=1)')
    ax4.set_xlabel('RFE Ranking Threshold', fontsize=14)
    ax4.set_ylabel('Cumulative Features Selected', fontsize=14)
    ax4.set_title('Cumulative Feature Selection by Ranking', fontsize=16, pad=20)
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    # Save plot
    importance_plot_path = os.path.join(BASE_DIR, 'feature_selection', 'rfe_feature_analysis.png')
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(importance_plot_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    # ================== SEPARATE DETAILED IMPORTANCE PLOT ==================
    plt.figure(figsize=(15, 10))

    # Sort features by importance for better visualization
    sorted_idx = np.argsort(top_importance)[::-1]
    sorted_importance = top_importance[sorted_idx]
    sorted_names = [top_names[i] for i in sorted_idx]

    y_pos = np.arange(len(sorted_names))
    colors = plt.cm.plasma(np.linspace(0, 1, len(sorted_names)))

    bars = plt.barh(y_pos, sorted_importance, color=colors, edgecolor='black', linewidth=0.5)
    plt.yticks(y_pos, sorted_names, fontsize=10)
    plt.xlabel('Feature Importance Score', fontsize=16)
    plt.ylabel('Features', fontsize=16)
    plt.title(f'RFE Selected Features Ranked by Importance\n(Top {len(sorted_names)} Features)',
              fontsize=18, pad=20)
    plt.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', ha='left', va='center', fontsize=9, weight='bold')

    # Add statistics box
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {np.mean(sorted_importance):.4f}\n'
    stats_text += f'Std: {np.std(sorted_importance):.4f}\n'
    stats_text += f'Min: {np.min(sorted_importance):.4f}\n'
    stats_text += f'Max: {np.max(sorted_importance):.4f}'

    plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
             verticalalignment='bottom', horizontalalignment='right',
             fontsize=12, weight='bold')

    plt.tight_layout()

    detailed_plot_path = os.path.join(BASE_DIR, 'feature_selection', 'rfe_detailed_importance.png')
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(detailed_plot_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"RFE analysis plots saved to: {importance_plot_path}")
    print(f"RFE detailed importance plot saved to: {detailed_plot_path}")


def perform_mig_feature_selection(X_train, y_train, X_val, y_val,
                                  feature_names, n_features=2000):
    """
    Perform Mutual Information Gain (MiG) based feature selection
    Select top n_features COLUMNS (features) from the dataset using Mutual Information
    """
    print("=" * 60)
    print("PERFORMING MUTUAL INFORMATION GAIN (MiG) FEATURE SELECTION")
    print("=" * 60)

    print(f"Original number of FEATURES (columns): {X_train.shape[1]}")
    print(f"Number of samples (rows): {X_train.shape[0]}")
    print(f"Target number of FEATURES to select: {n_features}")

    # If we already have fewer features than target, return all
    if X_train.shape[1] <= n_features:
        print(f"Dataset already has {X_train.shape[1]} features, which is <= {n_features}")
        selected_indices = np.arange(X_train.shape[1])
        selected_features = feature_names
        feature_importance = np.ones(X_train.shape[1])  # Dummy values
        return selected_indices, selected_features, feature_importance

    # Step 1: Calculate Mutual Information scores
    print("Computing Mutual Information scores for all features...")
    print("This may take a few minutes for large datasets...")

    # Use SelectKBest with mutual_info_classif
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)

    # Fit the selector with additional parameters for better MI estimation
    print("Fitting Mutual Information selector...")
    start_time = time.time()

    # Set random state for reproducibility in MI calculation
    X_train_selected = selector.fit_transform(X_train, y_train)

    mi_calculation_time = time.time() - start_time
    print(f"Mutual Information calculation completed in {mi_calculation_time:.2f} seconds")

    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    # Get MI scores
    mi_scores = selector.scores_

    print(f"Selected {len(selected_indices)} features (columns)")
    print(f"Selected feature indices range: {selected_indices.min()} to {selected_indices.max()}")
    print(f"Top 5 selected features: {selected_features[:5]}")

    # Create feature importance based on MI scores
    feature_importance = mi_scores.copy()

    # Print some statistics
    selected_mi_scores = mi_scores[selected_indices]

    print(f"Mutual Information score statistics for selected features:")
    print(f"  Mean MI score: {np.mean(selected_mi_scores):.6f}")
    print(f"  Std MI score: {np.std(selected_mi_scores):.6f}")
    print(f"  Min MI score: {np.min(selected_mi_scores):.6f}")
    print(f"  Max MI score: {np.max(selected_mi_scores):.6f}")
    print(f"  Median MI score: {np.median(selected_mi_scores):.6f}")

    # Analyze MI score distribution
    print(f"MI score distribution analysis:")
    print(f"  Features with MI > 0.1: {np.sum(selected_mi_scores > 0.1)}")
    print(f"  Features with MI > 0.05: {np.sum(selected_mi_scores > 0.05)}")
    print(f"  Features with MI > 0.01: {np.sum(selected_mi_scores > 0.01)}")
    print(f"  Features with MI = 0: {np.sum(selected_mi_scores == 0)}")

    # Step 2: Evaluate the selection
    # Train a simple logistic regression on selected features for evaluation
    X_val_selected = selector.transform(X_val)

    eval_model = LogisticRegression(random_state=42, max_iter=1000)
    eval_model.fit(X_train_selected, y_train)

    train_score = eval_model.score(X_train_selected, y_train)
    val_score = eval_model.score(X_val_selected, y_val)

    print(f"Logistic Regression on MiG selected features:")
    print(f"  Train Accuracy: {train_score:.4f}")
    print(f"  Validation Accuracy: {val_score:.4f}")

    # Step 3: Save MiG analysis results
    mig_results = {
        'selected_indices': selected_indices.tolist(),
        'selected_features': selected_features,
        'mi_scores': mi_scores.tolist(),
        'selected_mi_scores': selected_mi_scores.tolist(),
        'train_score': float(train_score),
        'val_score': float(val_score),
        'original_features': len(feature_names),
        'selected_features_count': len(selected_indices),
        'mi_calculation_time': float(mi_calculation_time),
        'features_mi_gt_0_1': int(np.sum(selected_mi_scores > 0.1)),
        'features_mi_gt_0_05': int(np.sum(selected_mi_scores > 0.05)),
        'features_mi_gt_0_01': int(np.sum(selected_mi_scores > 0.01)),
        'features_mi_eq_0': int(np.sum(selected_mi_scores == 0)),
        'mean_mi_score': float(np.mean(selected_mi_scores)),
        'std_mi_score': float(np.std(selected_mi_scores)),
        'min_mi_score': float(np.min(selected_mi_scores)),
        'max_mi_score': float(np.max(selected_mi_scores)),
        'median_mi_score': float(np.median(selected_mi_scores))
    }

    # Save MiG results
    mig_results_path = os.path.join(BASE_DIR, 'feature_selection', 'mig_results.json')
    with open(mig_results_path, 'w') as f:
        json.dump(mig_results, f, indent=2)

    # Save the MiG selector
    mig_selector_path = os.path.join(BASE_DIR, 'feature_selection', 'mig_selector.joblib')
    joblib.dump(selector, mig_selector_path)

    return selected_indices, selected_features, feature_importance


def compare_feature_selection_methods(X_train, y_train, X_val, y_val, feature_names, n_features=2000):
    """
    Compare RFE and MiG feature selection methods
    """
    print("\n" + "=" * 80)
    print("COMPARING FEATURE SELECTION METHODS: RFE vs MiG")
    print("=" * 80)

    results_comparison = {}

    # Test different RFE estimators
    rfe_estimators = ['logistic', 'rf']  # Removed 'svm' for speed

    for estimator_type in rfe_estimators:
        print(f"\n--- Testing RFE with {estimator_type} estimator ---")

        try:
            rfe_indices, rfe_features, rfe_importance = perform_rfe_feature_selection(
                X_train, y_train, X_val, y_val, feature_names,
                n_features=n_features, estimator_type=estimator_type
            )

            # Evaluate RFE selection
            X_train_rfe = X_train[:, rfe_indices]
            X_val_rfe = X_val[:, rfe_indices]

            eval_model = LogisticRegression(random_state=42, max_iter=1000)
            eval_model.fit(X_train_rfe, y_train)

            rfe_train_score = eval_model.score(X_train_rfe, y_train)
            rfe_val_score = eval_model.score(X_val_rfe, y_val)

            results_comparison[f'RFE_{estimator_type}'] = {
                'train_score': rfe_train_score,
                'val_score': rfe_val_score,
                'selected_features': len(rfe_indices),
                'feature_indices': rfe_indices,
                'method': f'RFE with {estimator_type}'
            }

            print(f"RFE ({estimator_type}) - Train: {rfe_train_score:.4f}, Val: {rfe_val_score:.4f}")

        except Exception as e:
            print(f"Error with RFE {estimator_type}: {e}")
            continue

    # Test MiG
    print(f"\n--- Testing MiG feature selection ---")
    try:
        mig_indices, mig_features, mig_importance = perform_mig_feature_selection(
            X_train, y_train, X_val, y_val, feature_names, n_features=n_features
        )

        # Evaluate MiG selection
        X_train_mig = X_train[:, mig_indices]
        X_val_mig = X_val[:, mig_indices]

        eval_model = LogisticRegression(random_state=42, max_iter=1000)
        eval_model.fit(X_train_mig, y_train)

        mig_train_score = eval_model.score(X_train_mig, y_train)
        mig_val_score = eval_model.score(X_val_mig, y_val)

        results_comparison['MiG'] = {
            'train_score': mig_train_score,
            'val_score': mig_val_score,
            'selected_features': len(mig_indices),
            'feature_indices': mig_indices,
            'method': 'Mutual Information Gain'
        }

        print(f"MiG - Train: {mig_train_score:.4f}, Val: {mig_val_score:.4f}")

    except Exception as e:
        print(f"Error with MiG: {e}")

    # Print comparison results
    print(f"\n" + "=" * 60)
    print("FEATURE SELECTION COMPARISON RESULTS")
    print("=" * 60)

    # Create comparison table
    comparison_df = pd.DataFrame(results_comparison).T
    comparison_df = comparison_df.round(4)
    print(comparison_df)

    # Find best method based on validation score
    best_method = max(results_comparison.items(), key=lambda x: x[1]['val_score'])
    print(f"\nBest method: {best_method[0]} with validation score: {best_method[1]['val_score']:.4f}")

    # Save comparison results
    comparison_path = os.path.join(BASE_DIR, 'feature_selection', 'method_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(results_comparison, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    # Plot comparison
    plot_feature_selection_comparison(results_comparison)

    return results_comparison, best_method


def plot_feature_selection_comparison(results_comparison):
    """Plot comparison of different feature selection methods"""

    methods = list(results_comparison.keys())
    train_scores = [results_comparison[method]['train_score'] for method in methods]
    val_scores = [results_comparison[method]['val_score'] for method in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot comparison
    x = np.arange(len(methods))
    width = 0.35

    ax1.bar(x - width / 2, train_scores, width, label='Training Score', alpha=0.8, color='skyblue')
    ax1.bar(x + width / 2, val_scores, width, label='Validation Score', alpha=0.8, color='lightcoral')

    ax1.set_xlabel('Feature Selection Method', fontsize=12)
    ax1.set_ylabel('Accuracy Score', fontsize=12)
    ax1.set_title('Feature Selection Methods Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Add value labels on bars
    for i, (train, val) in enumerate(zip(train_scores, val_scores)):
        ax1.text(i - width / 2, train + 0.01, f'{train:.3f}', ha='center', va='bottom', fontsize=10)
        ax1.text(i + width / 2, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Overfitting analysis (difference between train and val)
    overfitting = [train - val for train, val in zip(train_scores, val_scores)]

    colors = ['green' if diff < 0.05 else 'orange' if diff < 0.1 else 'red' for diff in overfitting]
    bars = ax2.bar(methods, overfitting, color=colors, alpha=0.7)

    ax2.set_xlabel('Feature Selection Method', fontsize=12)
    ax2.set_ylabel('Overfitting (Train - Val)', fontsize=12)
    ax2.set_title('Overfitting Analysis', fontsize=14)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add value labels
    for bar, diff in zip(bars, overfitting):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + (0.005 if height >= 0 else -0.01),
                 f'{diff:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

    # Add legend for overfitting colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Low overfitting (<0.05)'),
        Patch(facecolor='orange', label='Medium overfitting (0.05-0.1)'),
        Patch(facecolor='red', label='High overfitting (>0.1)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    comparison_plot_path = os.path.join(BASE_DIR, 'feature_selection', 'methods_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(comparison_plot_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature selection comparison plot saved to: {comparison_plot_path}")


# Perform Feature Selection Comparison
print("\n" + "=" * 60)
print("STARTING FEATURE SELECTION COMPARISON")
print("=" * 60)

# Compare different methods and select the best one
comparison_results, best_method_info = compare_feature_selection_methods(
    X_train_initial, y_train_initial, X_val_initial, y_val_initial,
    feature_names, n_features=2000
)

# Get the best method's selected features
best_method_name = best_method_info[0]
best_selected_indices = best_method_info[1]['feature_indices']

print(f"\nUsing best method: {best_method_name}")
print(f"Selected {len(best_selected_indices)} features")

# Apply feature selection to all datasets
print("\nApplying best feature selection to datasets...")
X_train = X_train_initial[:, best_selected_indices]
X_val = X_val_initial[:, best_selected_indices]
X_test = X_test_initial[:, best_selected_indices]

# Update selected features information
selected_indices = best_selected_indices
selected_features = [feature_names[i] for i in selected_indices]

print(f"Training set after feature selection: {X_train.shape}")
print(f"Validation set after feature selection: {X_val.shape}")
print(f"Test set after feature selection: {X_test.shape}")

# Update num_features for the model
num_features = X_train.shape[1]

print(f"Final number of features for AttBiLSTM: {num_features}")
print(f"Feature selection method used: {best_method_name}")


# Advanced Attention Mechanism
class MultiScaleAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiScaleAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Multiple attention heads with different perspectives
        self.query_nets = nn.ModuleList([
            nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)
        ])
        self.key_nets = nn.ModuleList([
            nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)
        ])
        self.value_nets = nn.ModuleList([
            nn.Linear(hidden_dim, self.head_dim) for _ in range(num_heads)
        ])

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Multi-head attention
        attention_outputs = []
        attention_weights_list = []

        for i in range(self.num_heads):
            query = self.query_nets[i](x)  # [batch, seq_len, head_dim]
            key = self.key_nets[i](x)
            value = self.value_nets[i](x)

            # Compute attention scores
            scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights_list.append(attention_weights)

            # Apply attention to values
            attended = torch.matmul(attention_weights, value)
            attention_outputs.append(attended)

        # Concatenate all heads
        multi_head_output = torch.cat(attention_outputs, dim=-1)

        # Project and apply residual connection
        output = self.output_proj(multi_head_output)
        output = self.dropout(output)
        output = self.layer_norm(x + output)  # Residual connection

        return output, attention_weights_list


# Custom activation functions
class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def forward(self, x):
        return x * torch.sigmoid(x)


# Register custom activations
nn.Mish = Mish
nn.Swish = Swish


# Advanced AttBiLSTM Model
class AdvancedAttBiLSTM(nn.Module):
    """Advanced Attention-based Bidirectional LSTM with custom activation functions"""

    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super(AdvancedAttBiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),  # GELU activation instead of ReLU
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = hidden_dim if i == 0 else hidden_dim * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                    dropout=dropout if i < num_layers - 1 else 0
                )
            )

        # Multi-scale attention mechanism
        self.attention = MultiScaleAttention(hidden_dim * 2, num_heads=4)

        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(),  # Mish activation function
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Swish(),  # Swish activation function
            nn.Dropout(dropout // 2)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 4, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, x):
        batch_size = x.size(0)

        # Project input features
        x = self.input_projection(x)  # [batch_size, hidden_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim] - create sequence dimension

        # Apply LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)  # [batch_size, seq_len, hidden_dim * 2]

        # Apply multi-scale attention
        x_attended, attention_weights = self.attention(x)  # [batch_size, seq_len, hidden_dim * 2]

        # Global average pooling and max pooling
        avg_pool = torch.mean(x_attended, dim=1)  # [batch_size, hidden_dim * 2]
        max_pool, _ = torch.max(x_attended, dim=1)  # [batch_size, hidden_dim * 2]

        # Combine pooled features
        combined_features = avg_pool + max_pool  # Element-wise addition

        # Feature fusion
        fused_features = self.feature_fusion(combined_features)

        # Final classification
        output = self.classifier(fused_features)

        return output, attention_weights


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create PyTorch data loaders"""
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, epochs=100, patience=15):
    """
    Train the AttBiLSTM model with early stopping

    Args:
        model: The AttBiLSTM model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of epochs to train
        patience: Number of epochs to wait before early stopping

    Returns:
        trained_model: The trained model
        history: Dictionary containing training history
        training_time: Total training time in seconds
    """

    # Move model to device
    model.to(device)

    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train_initial)
    class_weights = torch.FloatTensor([len(y_train_initial) / (len(class_counts) * count)
                                       for count in class_counts]).to(device)

    print(f"Class weights: {class_weights}")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with different learning rates for different parts of the model
    optimizer = optim.AdamW([
        {'params': model.input_projection.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': model.lstm_layers.parameters(), 'lr': 5e-4, 'weight_decay': 1e-5},
        {'params': model.attention.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': model.feature_fusion.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}
    ])

    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-6
    )

    # Mixed precision training (only for CUDA)
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for faster training")

    # Training history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    print("Starting training...")
    print(f"Total epochs: {epochs}, Early stopping patience: {patience}")
    print(f"Device: {device}")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        # ================== TRAINING PHASE ==================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batches = 0

        # Progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')

        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision if available
            if use_amp:
                with autocast():
                    output, attention_weights = model(data)
                    loss = criterion(output, target)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step with scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without mixed precision
                output, attention_weights = model(data)
                loss = criterion(output, target)

                # Backward pass
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                optimizer.step()

            # Calculate training metrics
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            train_batches += 1

            # Update progress bar
            current_acc = 100. * train_correct / train_total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # ================== VALIDATION PHASE ==================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0

        # Disable gradient computation for validation
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')

            for data, target in val_pbar:
                # Move data to device
                data, target = data.to(device), target.to(device)

                # Forward pass
                output, attention_weights = model(data)
                loss = criterion(output, target)

                # Calculate validation metrics
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                val_batches += 1

                # Update progress bar
                current_val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_val_acc:.2f}%'
                })

        # ================== EPOCH SUMMARY ==================
        # Calculate average metrics for the epoch
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update training history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)

        # Learning rate scheduling based on validation loss
        scheduler.step(avg_val_loss)

        # Print epoch summary
        print(f'\nEpoch {epoch + 1}/{epochs} Summary:')
        print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {current_lr:.2e}')

        # ================== EARLY STOPPING CHECK ==================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            # Save the best model state
            best_model_state = {
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }

            print(f'  *** New best validation loss: {best_val_loss:.4f} ***')
        else:
            patience_counter += 1
            print(f'  No improvement in validation loss. Patience: {patience_counter}/{patience}')

        # Check if we should stop early
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs!')
            print(f'Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}')
            break

        # Additional stopping condition: if learning rate becomes too small
        if current_lr < 1e-6:
            print(f'\nStopping training: Learning rate too small ({current_lr:.2e})')
            break

        print("-" * 60)

    # ================== TRAINING COMPLETION ==================
    total_training_time = time.time() - start_time

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f'\nLoaded best model from epoch {best_model_state["epoch"]}')
        print(f'Best validation loss: {best_model_state["best_val_loss"]:.4f}')
        print(f'Best validation accuracy: {best_model_state["val_acc"]:.2f}%')
    else:
        print('\nWarning: No improvement found during training!')

    print(f'\nTraining completed in {total_training_time:.2f} seconds')
    print(f'Average time per epoch: {total_training_time / (epoch + 1):.2f} seconds')

    # ================== FINAL STATISTICS ==================
    print("\nTraining Statistics:")
    print(f"  Total epochs trained: {epoch + 1}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Final train accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"  Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"  Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"  Final learning rate: {history['learning_rates'][-1]:.2e}")

    # Calculate some additional metrics
    train_loss_improvement = history['train_loss'][0] - history['train_loss'][-1]
    val_loss_improvement = history['val_loss'][0] - min(history['val_loss'])

    print(f"  Training loss improvement: {train_loss_improvement:.4f}")
    print(f"  Validation loss improvement: {val_loss_improvement:.4f}")

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return model, history, total_training_time


def calculate_detailed_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate comprehensive metrics including TP, TN, FP, FN, TPR, TNR, FPR, FNR

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities

    Returns:
        detailed_metrics: Dictionary with all calculated metrics
        cm: Confusion matrix
        roc_data: ROC curve data
    """

    print("Calculating detailed performance metrics...")

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # Determine if binary or multiclass
    num_classes = len(np.unique(y_true))
    print(f"Number of classes detected: {num_classes}")

    if num_classes == 2:
        # ================== BINARY CLASSIFICATION ==================
        tn, fp, fn, tp = cm.ravel()

        print(f"Binary Classification Metrics:")
        print(f"  True Positives (TP): {tp}")
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")

        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall/True Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity/True Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Fall-out/False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss rate/False Negative Rate

        # Additional binary metrics
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

        print(f"  Sensitivity (TPR/SN): {tpr:.4f}")
        print(f"  Specificity (TNR/SP): {tnr:.4f}")
        print(f"  False Positive Rate (FPR): {fpr:.4f}")
        print(f"  False Negative Rate (FNR): {fnr:.4f}")
        print(f"  Positive Predictive Value (PPV): {ppv:.4f}")
        print(f"  Negative Predictive Value (NPV): {npv:.4f}")

        # ROC AUC calculation
        try:
            fpr_roc, tpr_roc, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
            auc_score = auc(fpr_roc, tpr_roc)
            print(f"  AUC: {auc_score:.4f}")
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            fpr_roc, tpr_roc, auc_score = [0, 1], [0, 1], 0.5

        detailed_metrics = {
            'ACC': acc,
            'AUC': auc_score,
            'PRE': precision,
            'SP': tnr,  # Specificity
            'SN': tpr,  # Sensitivity
            'F1': f1,
            'MCC': mcc,
            'TPR': tpr,
            'FPR': fpr,
            'TNR': tnr,
            'FNR': fnr,
            'PPV': ppv,
            'NPV': npv,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }

        roc_data = (fpr_roc, tpr_roc, auc_score)

    else:
        # ================== MULTICLASS CLASSIFICATION ==================
        print(f"Multiclass Classification Metrics:")

        # Calculate macro-averaged rates
        tpr_list, fpr_list, tnr_list, fnr_list = [], [], [], []
        tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0

        auc_scores = []
        roc_curves = []

        for i in range(num_classes):
            # One-vs-Rest for each class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)

            # Calculate confusion matrix elements for this class
            tn_i = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp_i = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn_i = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            tp_i = np.sum((y_true_binary == 1) & (y_pred_binary == 1))

            # Accumulate totals
            tp_total += tp_i
            tn_total += tn_i
            fp_total += fp_i
            fn_total += fn_i

            # Calculate rates for this class
            tpr_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
            tnr_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
            fpr_i = fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0
            fnr_i = fn_i / (fn_i + tp_i) if (fn_i + tp_i) > 0 else 0

            tpr_list.append(tpr_i)
            tnr_list.append(tnr_i)
            fpr_list.append(fpr_i)
            fnr_list.append(fnr_i)

            print(f"  Class {i} - TPR: {tpr_i:.4f}, TNR: {tnr_i:.4f}, FPR: {fpr_i:.4f}, FNR: {fnr_i:.4f}")

            # ROC curve for each class
            try:
                fpr_roc, tpr_roc, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
                auc_i = auc(fpr_roc, tpr_roc)
                auc_scores.append(auc_i)
                roc_curves.append((fpr_roc, tpr_roc, auc_i))
                print(f"  Class {i} AUC: {auc_i:.4f}")
            except Exception as e:
                print(f"Warning: Could not calculate AUC for class {i}: {e}")
                auc_scores.append(0.5)
                roc_curves.append(([0, 1], [0, 1], 0.5))

        # Calculate macro averages
        mean_auc = np.mean(auc_scores)
        print(f"  Mean AUC: {mean_auc:.4f}")

        detailed_metrics = {
            'ACC': acc,
            'AUC': mean_auc,
            'PRE': precision,
            'SP': np.mean(tnr_list),  # Specificity
            'SN': np.mean(tpr_list),  # Sensitivity
            'F1': f1,
            'MCC': mcc,
            'TPR': np.mean(tpr_list),
            'FPR': np.mean(fpr_list),
            'TNR': np.mean(tnr_list),
            'FNR': np.mean(fnr_list),
            'TP': int(tp_total),
            'TN': int(tn_total),
            'FP': int(fp_total),
            'FN': int(fn_total)
        }

        roc_data = roc_curves

    # Print summary of key metrics
    print(f"\nSummary of Key Metrics:")
    print(f"  Accuracy: {detailed_metrics['ACC']:.4f}")
    print(f"  AUC: {detailed_metrics['AUC']:.4f}")
    print(f"  Precision: {detailed_metrics['PRE']:.4f}")
    print(f"  Sensitivity: {detailed_metrics['SN']:.4f}")
    print(f"  Specificity: {detailed_metrics['SP']:.4f}")
    print(f"  F1-Score: {detailed_metrics['F1']:.4f}")
    print(f"  MCC: {detailed_metrics['MCC']:.4f}")

    return detailed_metrics, cm, roc_data


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix with enhanced visualization

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """

    print(f"Creating confusion matrix plot...")

    plt.figure(figsize=(10, 8))

    # Create heatmap with customized appearance
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                annot_kws={"size": 20, "weight": "bold"})

    plt.title('Confusion Matrix of AttBiLSTM with RFE Feature Selection',
              fontsize=28, pad=20, weight='bold')
    plt.xlabel('Predicted Label', fontsize=24, weight='bold')
    plt.ylabel('True Label', fontsize=24, weight='bold')
    plt.xticks(fontsize=18, weight='bold')
    plt.yticks(fontsize=18, weight='bold')

    # Enhance aesthetics
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.4f}',
             transform=ax.transAxes, ha='center', fontsize=16, weight='bold')

    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Confusion matrix saved to: {save_path}")


def plot_training_curves(history, save_path):
    """
    Plot training and validation curves (loss and accuracy)

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """

    print(f"Creating training curves plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    # ================== LOSS CURVES ==================
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=3, label='Training Loss', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=3, label='Validation Loss', marker='s', markersize=4)
    ax1.set_title('Training and Validation Loss', fontsize=20, pad=20, weight='bold')
    ax1.set_xlabel('Epoch', fontsize=16, weight='bold')
    ax1.set_ylabel('Loss', fontsize=16, weight='bold')
    ax1.legend(fontsize=14, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)

    # Add min loss annotations
    min_train_loss = min(history['train_loss'])
    min_val_loss = min(history['val_loss'])
    min_train_epoch = history['train_loss'].index(min_train_loss) + 1
    min_val_epoch = history['val_loss'].index(min_val_loss) + 1

    ax1.annotate(f'Min Train: {min_train_loss:.4f}',
                 xy=(min_train_epoch, min_train_loss),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                 fontsize=10, weight='bold')

    ax1.annotate(f'Min Val: {min_val_loss:.4f}',
                 xy=(min_val_epoch, min_val_loss),
                 xytext=(10, -15), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                 fontsize=10, weight='bold')

    # ================== ACCURACY CURVES ==================
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=3, label='Training Accuracy', marker='o', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-', linewidth=3, label='Validation Accuracy', marker='s', markersize=4)
    ax2.set_title('Training and Validation Accuracy', fontsize=20, pad=20, weight='bold')
    ax2.set_xlabel('Epoch', fontsize=16, weight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=16, weight='bold')
    ax2.legend(fontsize=14, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)

    # Add max accuracy annotations
    max_train_acc = max(history['train_acc'])
    max_val_acc = max(history['val_acc'])
    max_train_epoch = history['train_acc'].index(max_train_acc) + 1
    max_val_epoch = history['val_acc'].index(max_val_acc) + 1

    ax2.annotate(f'Max Train: {max_train_acc:.2f}%',
                 xy=(max_train_epoch, max_train_acc),
                 xytext=(10, -15), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3),
                 fontsize=10, weight='bold')

    ax2.annotate(f'Max Val: {max_val_acc:.2f}%',
                 xy=(max_val_epoch, max_val_acc),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                 fontsize=10, weight='bold')

    # Enhance aesthetics for both subplots
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Training curves saved to: {save_path}")


def plot_training_loss(history, save_path):
    """
    Plot and save training and validation loss curve

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """

    print(f"Creating training loss plot...")

    plt.figure(figsize=(12, 8))

    epochs = range(1, len(history['train_loss']) + 1)

    plt.plot(epochs, history['train_loss'], 'b-', linewidth=3, label='Training Loss', marker='o', markersize=6)
    plt.plot(epochs, history['val_loss'], 'r-', linewidth=3, label='Validation Loss', marker='s', markersize=6)

    plt.title('Training and Validation Loss', fontsize=28, pad=20, weight='bold')
    plt.xlabel('Epoch', fontsize=24, weight='bold')
    plt.ylabel('Loss', fontsize=24, weight='bold')
    plt.legend(fontsize=20, loc='upper right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)

    # Add statistics box
    min_train_loss = min(history['train_loss'])
    min_val_loss = min(history['val_loss'])
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]

    stats_text = f'Min Train Loss: {min_train_loss:.4f}\n'
    stats_text += f'Min Val Loss: {min_val_loss:.4f}\n'
    stats_text += f'Final Train Loss: {final_train_loss:.4f}\n'
    stats_text += f'Final Val Loss: {final_val_loss:.4f}'

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
             fontsize=14, weight='bold')

    # Enhance aesthetics
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Training loss plot saved to: {save_path}")


def plot_training_accuracy(history, save_path):
    """
    Plot and save training and validation accuracy curve

    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """

    print(f"Creating training accuracy plot...")

    plt.figure(figsize=(12, 8))

    epochs = range(1, len(history['train_acc']) + 1)

    plt.plot(epochs, history['train_acc'], 'b-', linewidth=3, label='Training Accuracy', marker='o', markersize=6)
    plt.plot(epochs, history['val_acc'], 'r-', linewidth=3, label='Validation Accuracy', marker='s', markersize=6)

    plt.title('Training and Validation Accuracy', fontsize=28, pad=20, weight='bold')
    plt.xlabel('Epoch', fontsize=24, weight='bold')
    plt.ylabel('Accuracy (%)', fontsize=24, weight='bold')
    plt.legend(fontsize=20, loc='lower right')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.3)

    # Add statistics box
    max_train_acc = max(history['train_acc'])
    max_val_acc = max(history['val_acc'])
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]

    stats_text = f'Max Train Acc: {max_train_acc:.2f}%\n'
    stats_text += f'Max Val Acc: {max_val_acc:.2f}%\n'
    stats_text += f'Final Train Acc: {final_train_acc:.2f}%\n'
    stats_text += f'Final Val Acc: {final_val_acc:.2f}%'

    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes,
             verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
             fontsize=14, weight='bold')

    # Enhance aesthetics
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Training accuracy plot saved to: {save_path}")


def plot_roc_curve(roc_data, num_classes, save_path):
    """
    Plot ROC curves for binary or multiclass classification

    Args:
        roc_data: ROC curve data
        num_classes: Number of classes
        save_path: Path to save the plot
    """

    print(f"Creating ROC curve plot...")

    plt.figure(figsize=(10, 8))

    if num_classes == 2:
        # ================== BINARY CLASSIFICATION ==================
        fpr, tpr, auc_score = roc_data
        plt.plot(fpr, tpr, 'b-', linewidth=4,
                 label=f'ROC Curve (AUC = {auc_score:.4f})')

        # Add confidence interval (approximate)
        n_samples = len(fpr)
        std_auc = np.sqrt(auc_score * (1 - auc_score) / n_samples)
        plt.fill_between(fpr, np.maximum(tpr - std_auc, 0), np.minimum(tpr + std_auc, 1),
                         alpha=0.2, color='blue', label=f'±1 std (≈{std_auc:.3f})')

    else:
        # ================== MULTICLASS CLASSIFICATION ==================
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        mean_auc = 0

        for i, (fpr, tpr, auc_score) in enumerate(roc_data):
            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, color=color, linewidth=3,
                     label=f'Class {i} (AUC = {auc_score:.4f})')
            mean_auc += auc_score

        mean_auc /= len(roc_data)
        plt.plot([], [], 'k-', linewidth=4, label=f'Mean AUC = {mean_auc:.4f}')

    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=24, weight='bold')
    plt.ylabel('True Positive Rate', fontsize=24, weight='bold')
    plt.title('ROC Curves - AttBiLSTM with RFE Feature Selection', fontsize=28, pad=20, weight='bold')
    plt.legend(loc="lower right", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Add performance indicators
    if num_classes == 2:
        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = optimal_idx / len(fpr) if len(fpr) > 0 else 0.5
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                 label=f'Optimal Point (Threshold≈{optimal_threshold:.3f})')

    # Enhance aesthetics
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"ROC curve plot saved to: {save_path}")


def save_model_architecture(model, file_path):
    """
    Save detailed model architecture as text file

    Args:
        model: The trained model
        file_path: Path to save the architecture file
    """

    print(f"Saving model architecture...")

    with open(file_path, 'w') as f:
        f.write("Advanced AttBiLSTM Model Architecture with RFE Feature Selection\n")
        f.write("=" * 80 + "\n\n")

        # ================== MODEL CONFIGURATION ==================
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Input Dimension: {model.input_dim}\n")
        f.write(f"Hidden Dimension: {model.hidden_dim}\n")
        f.write(f"Number of Classes: {model.num_classes}\n")
        f.write(f"Number of LSTM Layers: {model.num_layers}\n\n")

        # ================== FEATURE SELECTION INFO ==================
        f.write("FEATURE SELECTION:\n")
        f.write("-" * 40 + "\n")
        if 'feature_names' in globals():
            f.write(f"Original Features: {len(feature_names)}\n")
        f.write(f"Selected Features: {model.input_dim}\n")
        if 'selected_features' in globals():
            f.write(f"Selected Features Count: {len(selected_features)}\n")
            reduction_ratio = (1 - model.input_dim / len(feature_names)) * 100
            f.write(f"Feature Reduction: {reduction_ratio:.2f}%\n")
        f.write(f"Feature Selection Method: {best_method_name}\n\n")

        # ================== MODEL ARCHITECTURE ==================
        f.write("DETAILED MODEL ARCHITECTURE:\n")
        f.write("-" * 40 + "\n")
        f.write(str(model))
        f.write("\n\n")

        # ================== PARAMETER COUNT ==================
        f.write("PARAMETER ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {non_trainable_params:,}\n")
        f.write(f"Model Size (MB): {total_params * 4 / (1024 * 1024):.2f}\n\n")

        # ================== LAYER-WISE PARAMETERS ==================
        f.write("LAYER-WISE PARAMETER COUNT:\n")
        f.write("-" * 40 + "\n")

        layer_params = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    layer_params[name] = params

        # Sort by parameter count
        sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)

        for name, params in sorted_layers:
            percentage = (params / total_params) * 100
            f.write(f"{name}: {params:,} parameters ({percentage:.2f}%)\n")

        f.write("\n")

        # ================== MODEL COMPONENTS ==================
        f.write("MODEL COMPONENTS BREAKDOWN:\n")
        f.write("-" * 40 + "\n")

        component_params = {
            'Input Projection': sum(p.numel() for p in model.input_projection.parameters()),
            'LSTM Layers': sum(p.numel() for p in model.lstm_layers.parameters()),
            'Attention Mechanism': sum(p.numel() for p in model.attention.parameters()),
            'Feature Fusion': sum(p.numel() for p in model.feature_fusion.parameters()),
            'Classifier': sum(p.numel() for p in model.classifier.parameters())
        }

        for component, params in component_params.items():
            percentage = (params / total_params) * 100
            f.write(f"{component}: {params:,} parameters ({percentage:.2f}%)\n")

        f.write("\n")

        # ================== ACTIVATION FUNCTIONS ==================
        f.write("ACTIVATION FUNCTIONS USED:\n")
        f.write("-" * 40 + "\n")
        f.write("- GELU (Gaussian Error Linear Unit)\n")
        f.write("- Mish (x * tanh(softplus(x)))\n")
        f.write("- Swish (x * sigmoid(x))\n")
        f.write("- Softmax (for attention and output)\n\n")

        # ================== REGULARIZATION TECHNIQUES ==================
        f.write("REGULARIZATION TECHNIQUES:\n")
        f.write("-" * 40 + "\n")
        f.write("- Batch Normalization\n")
        f.write("- Dropout (various rates)\n")
        f.write("- Gradient Clipping (max_norm=1.0)\n")
        f.write("- Weight Decay in optimizer\n")
        f.write("- Early Stopping\n\n")

        # ================== TRAINING CONFIGURATION ==================
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write("- Optimizer: AdamW with different learning rates\n")
        f.write("- Loss Function: CrossEntropyLoss with class weights\n")
        f.write("- Scheduler: ReduceLROnPlateau\n")
        f.write("- Mixed Precision: Enabled for CUDA devices\n")
        f.write("- Gradient Clipping: max_norm=1.0\n\n")

        # ================== DEVICE INFORMATION ==================
        f.write("DEVICE INFORMATION:\n")
        f.write("-" * 40 + "\n")
        if 'device' in globals():
            f.write(f"Training Device: {device}\n")
        f.write(f"Model Location: {next(model.parameters()).device}\n\n")

        # ================== TIMESTAMP ==================
        from datetime import datetime
        f.write("GENERATION INFO:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")

    print(f"Model architecture saved to: {file_path}")


def run_complete_analysis():
    """
    Run the complete AttBiLSTM analysis pipeline with RFE feature selection

    Returns:
        trained_model: The trained model
        detailed_metrics: Dictionary with all performance metrics
        history: Training history
    """

    print("=" * 80)
    print("STARTING COMPLETE AttBiLSTM ANALYSIS PIPELINE WITH RFE")
    print("=" * 80)

    start_time = time.time()

    # ================== SETUP AND INITIALIZATION ==================
    print("\n1. SETTING UP ANALYSIS ENVIRONMENT")
    print("-" * 50)

    # Validate required variables
    required_vars = ['X_train', 'y_train_initial', 'X_val', 'y_val_initial',
                     'X_test', 'y_test_initial', 'num_features', 'num_classes']

    for var in required_vars:
        if var not in globals():
            raise ValueError(f"Required variable '{var}' not found. Please run feature selection first.")

    print(f"✓ Dataset validated")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Validation samples: {X_val.shape[0]}")
    print(f"  - Test samples: {X_test.shape[0]}")
    print(f"  - Features: {num_features}")
    print(f"  - Classes: {num_classes}")
    print(f"  - Feature selection method: {best_method_name}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train_initial, X_val, y_val_initial, batch_size=32)

    print(f"✓ Data loaders created")
    print(f"  - Batch size: 32")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")

    # ================== MODEL INITIALIZATION ==================
    print("\n2. INITIALIZING MODEL")
    print("-" * 50)

    model = AdvancedAttBiLSTM(
        input_dim=num_features,
        num_classes=num_classes,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model initialized successfully")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: {total_params * 4 / (1024 * 1024):.2f} MB")

    # ================== MODEL TRAINING ==================
    print("\n3. TRAINING MODEL")
    print("-" * 50)

    trained_model, history, training_time = train_model(
        model, train_loader, val_loader, epochs=100, patience=15
    )

    print(f"✓ Model training completed")
    print(f"  - Training time: {training_time:.2f} seconds")
    print(f"  - Final training accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"  - Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"  - Best validation accuracy: {max(history['val_acc']):.2f}%")

    # ================== MODEL SAVING ==================
    print("\n4. SAVING MODEL")
    print("-" * 50)

    model_save_path = os.path.join(BASE_DIR, 'models', 'attbilstm_rfe_model.pth')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_config': {
            'input_dim': num_features,
            'num_classes': num_classes,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3
        },
        'training_history': history,
        'scaler_params': {
            'mean_': scaler.mean_.tolist() if 'scaler' in globals() else None,
            'scale_': scaler.scale_.tolist() if 'scaler' in globals() else None
        },
        'selected_features': selected_features if 'selected_features' in globals() else None,
        'selected_indices': selected_indices.tolist() if 'selected_indices' in globals() else None,
        'feature_selection_method': best_method_name,
        'comparison_results': comparison_results,
        'training_time': training_time,
        'timestamp': time.time()
    }, model_save_path)

    print(f"✓ Model saved to: {model_save_path}")

    # Save model architecture
    architecture_path = os.path.join(BASE_DIR, 'models', 'model_architecture.txt')
    save_model_architecture(trained_model, architecture_path)
    print(f"✓ Model architecture saved to: {architecture_path}")

    # ================== MODEL EVALUATION ==================
    print("\n5. EVALUATING MODEL ON TEST SET")
    print("-" * 50)

    trained_model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    test_start_time = time.time()
    with torch.no_grad():
        test_outputs, attention_weights = trained_model(X_test_tensor)
        test_probabilities = F.softmax(test_outputs, dim=1).cpu().numpy()
        test_predictions = np.argmax(test_probabilities, axis=1)
    testing_time = time.time() - test_start_time

    print(f"✓ Test predictions generated")
    print(f"  - Testing time: {testing_time:.4f} seconds")
    print(f"  - Inference speed: {len(X_test) / testing_time:.2f} samples/second")

    # Calculate detailed metrics
    detailed_metrics, confusion_mat, roc_data = calculate_detailed_metrics(
        y_test_initial, test_predictions, test_probabilities
    )

    # Add additional information to metrics
    detailed_metrics['Training Time'] = training_time
    detailed_metrics['Testing Time'] = testing_time
    detailed_metrics['Original Features'] = len(feature_names) if 'feature_names' in globals() else num_features
    detailed_metrics['Selected Features'] = num_features
    detailed_metrics['Feature Selection Method'] = best_method_name
    detailed_metrics['Model Parameters'] = total_params
    detailed_metrics['Inference Speed'] = len(X_test) / testing_time

    print(f"✓ Detailed metrics calculated")

    # ================== RESULTS SAVING ==================
    print("\n6. SAVING RESULTS")
    print("-" * 50)

    # Create results DataFrame
    results_df = pd.DataFrame([detailed_metrics])
    results_df = results_df.round(4)

    # Print main metrics
    print("\nKEY PERFORMANCE METRICS:")
    main_metrics = ['ACC', 'AUC', 'PRE', 'SP', 'SN', 'F1', 'MCC']
    for metric in main_metrics:
        if metric in detailed_metrics:
            print(f"  {metric}: {detailed_metrics[metric]:.4f}")

    print(f"\nTIMING METRICS:")
    print(f"  Training Time: {detailed_metrics['Training Time']:.2f} seconds")
    print(f"  Testing Time: {detailed_metrics['Testing Time']:.4f} seconds")
    print(f"  Inference Speed: {detailed_metrics['Inference Speed']:.2f} samples/second")

    print(f"\nFEATURE SELECTION METRICS:")
    print(f"  Original Features: {detailed_metrics['Original Features']}")
    print(f"  Selected Features: {detailed_metrics['Selected Features']}")
    print(
        f"  Feature Reduction: {(1 - detailed_metrics['Selected Features'] / detailed_metrics['Original Features']) * 100:.2f}%")
    print(f"  Selection Method: {detailed_metrics['Feature Selection Method']}")

    # Save results to CSV
    results_path = os.path.join(BASE_DIR, 'results', 'detailed_metrics.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Results saved to: {results_path}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(BASE_DIR, 'results', 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"✓ Training history saved to: {history_path}")

    # Save selected features
    if 'selected_features' in globals() and 'selected_indices' in globals():
        selected_features_df = pd.DataFrame({
            'feature_name': selected_features,
            'original_index': selected_indices,
            'selection_method': best_method_name
        })
        selected_features_path = os.path.join(BASE_DIR, 'feature_selection', 'selected_features.csv')
        selected_features_df.to_csv(selected_features_path, index=False)
        print(f"✓ Selected features saved to: {selected_features_path}")

    # Save feature selection comparison results
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_results_path = os.path.join(BASE_DIR, 'feature_selection', 'method_comparison.csv')
    comparison_df.to_csv(comparison_results_path)
    print(f"✓ Feature selection comparison saved to: {comparison_results_path}")

    # ================== GENERATING PLOTS ==================
    print("\n7. GENERATING VISUALIZATIONS")
    print("-" * 50)

    # 1. Confusion Matrix
    class_names = [f'Class {i}' for i in range(num_classes)]
    cm_path = os.path.join(BASE_DIR, 'plots', 'confusion_matrix.png')
    plot_confusion_matrix(confusion_mat, class_names, cm_path)

    # 2. Training Curves (combined)
    curves_path = os.path.join(BASE_DIR, 'plots', 'training_curves.png')
    plot_training_curves(history, curves_path)

    # 3. Individual training plots
    loss_path = os.path.join(BASE_DIR, 'plots', 'loss_curve.png')
    accuracy_path = os.path.join(BASE_DIR, 'plots', 'accuracy_curve.png')
    plot_training_loss(history, loss_path)
    plot_training_accuracy(history, accuracy_path)

    # 4. ROC Curves
    roc_path = os.path.join(BASE_DIR, 'plots', 'roc_curves.png')
    plot_roc_curve(roc_data, num_classes, roc_path)

    print(f"✓ All plots generated successfully")

    # ================== ADDITIONAL REPORTS ==================
    print("\n8. GENERATING ADDITIONAL REPORTS")
    print("-" * 50)

    # Save detailed classification report
    class_report = classification_report(y_test_initial, test_predictions,
                                         target_names=class_names,
                                         output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_path = os.path.join(BASE_DIR, 'results', 'classification_report.csv')
    class_report_df.to_csv(class_report_path)
    print(f"✓ Classification report saved to: {class_report_path}")

    # Save attention weights analysis
    if attention_weights:
        attention_analysis_path = os.path.join(BASE_DIR, 'results', 'attention_analysis.txt')
        with open(attention_analysis_path, 'w') as f:
            f.write("Attention Weights Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of attention heads: {len(attention_weights)}\n")
            f.write(f"Attention tensor shape: {attention_weights[0].shape}\n\n")

            # Calculate average attention across all samples and heads
            avg_attention = torch.mean(torch.stack(attention_weights), dim=0)
            f.write(f"Average attention statistics:\n")
            f.write(f"  Mean: {torch.mean(avg_attention).item():.6f}\n")
            f.write(f"  Std: {torch.std(avg_attention).item():.6f}\n")
            f.write(f"  Min: {torch.min(avg_attention).item():.6f}\n")
            f.write(f"  Max: {torch.max(avg_attention).item():.6f}\n\n")

            # Attention distribution analysis
            f.write("Attention Distribution Analysis:\n")
            attention_flat = avg_attention.flatten().cpu().numpy()
            f.write(f"  Median: {np.median(attention_flat):.6f}\n")
            f.write(f"  25th percentile: {np.percentile(attention_flat, 25):.6f}\n")
            f.write(f"  75th percentile: {np.percentile(attention_flat, 75):.6f}\n")

        print(f"✓ Attention analysis saved to: {attention_analysis_path}")

    # Create comprehensive summary report
    summary_path = os.path.join(BASE_DIR, 'results', 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("COMPREHENSIVE AttBiLSTM ANALYSIS SUMMARY WITH RFE\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {len(X_train) + len(X_val) + len(X_test)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Number of features: {num_features}\n")
        f.write(f"Number of classes: {num_classes}\n")
        if 'feature_names' in globals():
            f.write(f"Original features: {len(feature_names)}\n")
            f.write(f"Feature reduction: {(1 - num_features / len(feature_names)) * 100:.2f}%\n")
        f.write(f"Feature selection method: {best_method_name}\n")
        f.write("\n")

        f.write("FEATURE SELECTION COMPARISON:\n")
        f.write("-" * 40 + "\n")
        for method, results in comparison_results.items():
            f.write(f"{method}:\n")
            f.write(f"  Train Score: {results['train_score']:.4f}\n")
            f.write(f"  Val Score: {results['val_score']:.4f}\n")
            f.write(f"  Selected Features: {results['selected_features']}\n")
        f.write(f"Best method: {best_method_name}\n\n")

        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Architecture: Advanced AttBiLSTM\n")
        f.write(f"Hidden dimension: 128\n")
        f.write(f"LSTM layers: 2\n")
        f.write(f"Dropout rate: 0.3\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        for metric, value in detailed_metrics.items():
            if isinstance(value, (int, float)) and metric not in ['Original Features', 'Selected Features',
                                                                  'Model Parameters']:
                f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")

        f.write("TRAINING INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training epochs: {len(history['train_loss'])}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Final training accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%\n")
        f.write(f"Best validation accuracy: {max(history['val_acc']):.2f}%\n\n")

        f.write("CLASS DISTRIBUTION (TEST SET):\n")
        f.write("-" * 40 + "\n")
        unique_classes, class_counts = np.unique(y_test_initial, return_counts=True)
        for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
            percentage = (count / len(y_test_initial)) * 100
            f.write(f"Class {cls}: {count} samples ({percentage:.2f}%)\n")

        f.write("\n")
        f.write("ANALYSIS COMPLETED SUCCESSFULLY!\n")
        f.write(f"Total analysis time: {time.time() - start_time:.2f} seconds\n")

    print(f"✓ Comprehensive summary saved to: {summary_path}")

    # ================== FINAL SUMMARY ==================
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Total analysis time: {total_time:.2f} seconds")
    print(f"All results saved in directory: {BASE_DIR}")

    print(f"\nFILES GENERATED:")
    print(f"📁 Models:")
    print(f"  - attbilstm_rfe_model.pth")
    print(f"  - model_architecture.txt")
    print(f"📁 Results:")
    print(f"  - detailed_metrics.csv")
    print(f"  - training_history.csv")
    print(f"  - classification_report.csv")
    print(f"  - analysis_summary.txt")
    print(f"  - attention_analysis.txt")
    print(f"📁 Plots:")
    print(f"  - confusion_matrix.png/pdf")
    print(f"  - training_curves.png/pdf")
    print(f"  - loss_curve.png/pdf")
    print(f"  - accuracy_curve.png/pdf")
    print(f"  - roc_curves.png/pdf")
    print(f"📁 Feature Selection:")
    print(f"  - selected_features.csv")
    print(f"  - method_comparison.csv")
    print(f"  - rfe_results.json")
    print(f"  - mig_results.json")
    print(f"  - rfe_feature_analysis.png/pdf")
    print(f"  - rfe_detailed_importance.png/pdf")
    print(f"  - methods_comparison.png/pdf")

    print(f"\nKEY RESULTS:")
    print(f"🎯 Test Accuracy: {detailed_metrics['ACC']:.4f}")
    print(f"🎯 Test AUC: {detailed_metrics['AUC']:.4f}")
    print(f"🎯 Test F1-Score: {detailed_metrics['F1']:.4f}")
    print(f"🎯 Matthews Correlation: {detailed_metrics['MCC']:.4f}")
    print(f"🎯 Feature Selection Method: {best_method_name}")
    print(
        f"🎯 Features Reduced: {(1 - detailed_metrics['Selected Features'] / detailed_metrics['Original Features']) * 100:.1f}%")

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"\n✓ Memory cleanup completed")
    print("=" * 80)

    return trained_model, detailed_metrics, history


if __name__ == "__main__":
    try:
        final_model, final_metrics, final_history = run_complete_analysis()

        # Print final performance summary
        print("\n" + "=" * 60)
        print("FINAL PERFORMANCE SUMMARY WITH RFE")
        print("=" * 60)
        print(f"Accuracy (ACC): {final_metrics['ACC']:.4f}")
        print(f"AUC: {final_metrics['AUC']:.4f}")
        print(f"Precision (PRE): {final_metrics['PRE']:.4f}")
        print(f"Specificity (SP): {final_metrics['SP']:.4f}")
        print(f"Sensitivity (SN): {final_metrics['SN']:.4f}")
        print(f"F1-Score: {final_metrics['F1']:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {final_metrics['MCC']:.4f}")
        print(f"Training Time: {final_metrics['Training Time']:.2f} seconds")
        print(f"Testing Time: {final_metrics['Testing Time']:.4f} seconds")
        print(f"Feature Selection Method: {final_metrics['Feature Selection Method']}")
        print(
            f"Feature Reduction: {(1 - final_metrics['Selected Features'] / final_metrics['Original Features']) * 100:.2f}%")
        print(f"Selected Features: {final_metrics['Selected Features']} / {final_metrics['Original Features']}")

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        print("\nMemory cleanup completed.")
        print("\n🎉 RFE-enhanced AttBiLSTM analysis pipeline completed successfully!")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("Analysis pipeline finished.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_curve, auc, precision_score, f1_score,
                             matthews_corrcoef, accuracy_score, recall_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
import shap
import joblib


warnings.filterwarnings('ignore')

# Set font family globally
plt.rcParams['font.family'] = 'Times New Roman'
dpi = 1000
plt.rcParams['figure.dpi'] = dpi


# Base directory
BASE_DIR = "AttBiLSTM_Analysis_With_SHAP_FS"
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

# Initial split for SHAP feature selection
X_train_initial, X_temp, y_train_initial, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
X_val_initial, X_test_initial, y_val_initial, y_test_initial = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Initial training set: {X_train_initial.shape[0]} samples")
print(f"Initial validation set: {X_val_initial.shape[0]} samples")
print(f"Initial test set: {X_test_initial.shape[0]} samples")



def perform_shap_feature_selection(X_train, y_train, X_val, y_val,
                                   feature_names, n_features=2000):
    """
    Perform SHAP-based feature selection
    Select top n_features COLUMNS (features) from the dataset
    """
    print("=" * 60)
    print("PERFORMING SHAP FEATURE SELECTION")
    print("=" * 60)

    print(f"Original number of FEATURES (columns): {X_train.shape[1]}")
    print(f"Number of samples (rows): {X_train.shape[0]}")
    print(f"Target number of FEATURES to select: {n_features}")

    # If we already have fewer features than target, return all
    if X_train.shape[1] <= n_features:
        print(f"Dataset already has {X_train.shape[1]} features, which is <= {n_features}")
        selected_indices = np.arange(X_train.shape[1])
        selected_features = feature_names
        shap_values_mean = np.ones(X_train.shape[1])  # Dummy values
        return selected_indices, selected_features, shap_values_mean

    # Step 1: Train a Random Forest model for SHAP analysis
    print("Training Random Forest model for SHAP analysis...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    # Fit the model
    rf_model.fit(X_train, y_train)

    # Evaluate RF model
    train_score = rf_model.score(X_train, y_train)
    val_score = rf_model.score(X_val, y_val)
    print(f"Random Forest - Train Accuracy: {train_score:.4f}")
    print(f"Random Forest - Validation Accuracy: {val_score:.4f}")

    # Step 2: Calculate SHAP values
    print("Calculating SHAP values...")

    # Use a subset of training data for SHAP calculation to speed up
    sample_size = min(500, X_train.shape[0])  # Reduced sample size for speed
    sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    X_sample = X_train[sample_indices]

    print(f"Using {sample_size} samples for SHAP calculation (out of {X_train.shape[0]} total)")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)

    # Calculate SHAP values
    print(f"Computing SHAP values for {sample_size} samples and {X_train.shape[1]} features...")
    shap_values = explainer.shap_values(X_sample)

    print(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"SHAP values shape (list): {[sv.shape for sv in shap_values]}")
    else:
        print(f"SHAP values shape: {shap_values.shape}")

    # Step 3: Calculate feature importance scores
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # Multiclass case - shap_values is a list of arrays (one per class)
        print("Processing multiclass SHAP values...")
        # Take mean absolute SHAP values across all classes
        shap_values_combined = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        # Mean absolute SHAP value for each FEATURE across all samples
        feature_importance = np.mean(shap_values_combined, axis=0)
    else:
        # Binary case - check dimensions
        if len(shap_values.shape) == 3:
            # 3D array: (samples, features, classes)
            print("Processing 3D SHAP values (samples, features, classes)...")
            # For binary classification, we can use either class or take mean
            # Using class 1 (positive class) for feature importance
            shap_values_class1 = np.abs(shap_values[:, :, 1])  # Use positive class
            feature_importance = np.mean(shap_values_class1, axis=0)  # Mean across samples
        else:
            # 2D array: (samples, features)
            print("Processing 2D SHAP values (samples, features)...")
            shap_values_combined = np.abs(shap_values)
            feature_importance = np.mean(shap_values_combined, axis=0)

    print(f"Feature importance shape: {feature_importance.shape}")
    print(f"This should be ({X_train.shape[1]},) - one importance score per feature")

    # Step 4: Select top FEATURES (columns)
    print(f"Selecting top {n_features} FEATURES based on SHAP importance...")

    # Get indices of top features (highest importance first)
    selected_indices = np.argsort(feature_importance)[::-1][:n_features]
    selected_features = [feature_names[i] for i in selected_indices]

    print(f"Selected {len(selected_indices)} features (columns)")
    print(f"Selected feature indices range: {selected_indices.min()} to {selected_indices.max()}")
    print(f"Top 5 selected features: {selected_features[:5]}")

    # Step 5: Save SHAP analysis results
    shap_results = {
        'selected_indices': selected_indices.tolist(),
        'selected_features': selected_features,
        'feature_importance': feature_importance.tolist(),
        'rf_train_score': float(train_score),
        'rf_val_score': float(val_score),
        'original_features': len(feature_names),
        'selected_features_count': len(selected_indices)
    }

    # Save SHAP results
    shap_results_path = os.path.join(BASE_DIR, 'feature_selection', 'shap_results.json')
    with open(shap_results_path, 'w') as f:
        json.dump(shap_results, f, indent=2)

    # Save the Random Forest model
    rf_model_path = os.path.join(BASE_DIR, 'feature_selection', 'rf_model_for_shap.joblib')
    joblib.dump(rf_model, rf_model_path)

    # Create feature importance plot
    plot_shap_feature_importance(feature_importance, feature_names, selected_indices, n_features)

    print(f"SHAP analysis results saved to: {shap_results_path}")
    print(f"Random Forest model saved to: {rf_model_path}")

    return selected_indices, selected_features, feature_importance



def plot_shap_feature_importance(feature_importance, feature_names, selected_indices, n_features):
    """Plot SHAP feature importance"""

    # Plot top features
    top_indices = selected_indices[:min(50, n_features)]  # Show top 50 features
    top_importance = feature_importance[top_indices]
    top_names = [feature_names[int(i)] for i in top_indices]

    plt.figure(figsize=(12, 10))

    # Create horizontal bar plot
    y_pos = np.arange(len(top_names))
    bars = plt.barh(y_pos, top_importance, color='skyblue', edgecolor='navy', linewidth=0.5)

    plt.yticks(y_pos, top_names, fontsize=8)
    plt.xlabel('Mean |SHAP Value|', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title(f'Top {len(top_names)} Features by SHAP Importance', fontsize=16, pad=20)
    plt.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', ha='left', va='center', fontsize=8)

    plt.tight_layout()

    # Save plot
    importance_plot_path = os.path.join(BASE_DIR, 'feature_selection', 'shap_feature_importance.png')
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(importance_plot_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot distribution of all feature importance scores
    plt.figure(figsize=(10, 6))
    plt.hist(feature_importance, bins=50, color='lightblue', edgecolor='black', alpha=0.7)

    # Add threshold line only if we have enough features
    if len(selected_indices) > 0 and n_features <= len(feature_importance):
        threshold_idx = min(n_features - 1, len(selected_indices) - 1)
        threshold_value = feature_importance[selected_indices[threshold_idx]]
        plt.axvline(threshold_value, color='red', linestyle='--',
                    linewidth=2, label=f'Selection Threshold (Top {n_features})')
        plt.legend()

    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.title('Distribution of SHAP Feature Importance Scores', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    dist_plot_path = os.path.join(BASE_DIR, 'feature_selection', 'shap_importance_distribution.png')
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(dist_plot_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SHAP importance plots saved to: {importance_plot_path}")
    print(f"SHAP distribution plot saved to: {dist_plot_path}")



# Perform SHAP Feature Selection
print("\n" + "=" * 60)
print("STARTING SHAP FEATURE SELECTION")
print("=" * 60)

selected_indices, selected_features, feature_importance_scores = perform_shap_feature_selection(
    X_train_initial, y_train_initial, X_val_initial, y_val_initial,
    feature_names, n_features=2000
)

# Apply feature selection to all datasets
print("\nApplying SHAP feature selection to datasets...")
X_train = X_train_initial[:, selected_indices]
X_val = X_val_initial[:, selected_indices]
X_test = X_test_initial[:, selected_indices]

print(f"Training set after feature selection: {X_train.shape}")
print(f"Validation set after feature selection: {X_val.shape}")
print(f"Test set after feature selection: {X_test.shape}")

# Update num_features for the model
num_features = X_train.shape[1]

print(f"Final number of features for AttBiLSTM: {num_features}")



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
    """Train the AttBiLSTM model with early stopping"""

    model.to(device)

    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train_initial)
    class_weights = torch.FloatTensor([len(y_train_initial) / (len(class_counts) * count)
                                       for count in class_counts]).to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': model.input_projection.parameters(), 'lr': 1e-3},
        {'params': model.lstm_layers.parameters(), 'lr': 5e-4},
        {'params': model.attention.parameters(), 'lr': 1e-3},
        {'params': model.feature_fusion.parameters(), 'lr': 1e-3},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Mixed precision training
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print("Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    output, _ = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output, _ = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)

        # Calculate average losses and accuracies
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    training_time = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f'Training completed in {training_time:.2f} seconds')
    return model, history, training_time



def calculate_detailed_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics including TP, TN, FP, FN, TPR, TNR, FPR, FNR"""

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # For binary classification
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = cm.ravel()

        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Fall-out
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss rate

        # AUC
        fpr_roc, tpr_roc, _ = roc_curve(y_true, y_pred_proba[:, 1])
        auc_score = auc(fpr_roc, tpr_roc)

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
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }

        roc_data = (fpr_roc, tpr_roc, auc_score)

    else:
        # For multiclass classification
        # Calculate macro-averaged rates
        tpr_list, fpr_list, tnr_list, fnr_list = [], [], [], []
        tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0

        auc_scores = []
        roc_curves = []

        for i in range(len(np.unique(y_true))):
            # One-vs-Rest for each class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)

            tn_i = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fp_i = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn_i = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            tp_i = np.sum((y_true_binary == 1) & (y_pred_binary == 1))

            tp_total += tp_i
            tn_total += tn_i
            fp_total += fp_i
            fn_total += fn_i

            tpr_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
            tnr_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
            fpr_i = fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0
            fnr_i = fn_i / (fn_i + tp_i) if (fn_i + tp_i) > 0 else 0

            tpr_list.append(tpr_i)
            tnr_list.append(tnr_i)
            fpr_list.append(fpr_i)
            fnr_list.append(fnr_i)

            # ROC curve for each class
            fpr_roc, tpr_roc, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
            auc_i = auc(fpr_roc, tpr_roc)
            auc_scores.append(auc_i)
            roc_curves.append((fpr_roc, tpr_roc, auc_i))

        detailed_metrics = {
            'ACC': acc,
            'AUC': np.mean(auc_scores),
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

    return detailed_metrics, cm, roc_data



def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, annot_kws={"size": 20})

    plt.title('Confusion Matrix of AttBiLSTM with SHAP FS', fontsize=28, pad=20)
    plt.xlabel('Predicted Label', fontsize=24)
    plt.ylabel('True Label', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Enhance aesthetics
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
    plt.close()



def plot_training_curves(history, save_path):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss curves
    ax1.plot(history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Training and Validation Loss of AttBiLSTM with SHAP FS', fontsize=28, pad=20)
    ax1.set_xlabel('Epoch', fontsize=24)
    ax1.set_ylabel('Loss', fontsize=24)
    ax1.legend(fontsize=18)
    ax1.grid(False)

    # Accuracy curves
    ax2.plot(history['train_acc'], 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy of AttBiLSTM with SHAP FS', fontsize=28, pad=20)
    ax2.set_xlabel('Epoch', fontsize=24)
    ax2.set_ylabel('Accuracy (%)', fontsize=24)
    ax2.legend(fontsize=18)
    ax2.grid(False)

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
    plt.close()



def plot_training_loss(history, save_path):
    """Plot and save training and validation loss curve"""

    plt.figure(figsize=(10, 8))
    plt.plot(history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    plt.plot(history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss of AttBiLSTM with SHAP FS', fontsize=28, pad=20)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(False)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
    plt.close()



def plot_training_accuracy(history, save_path):
    """Plot and save training and validation accuracy curve"""

    plt.figure(figsize=(10, 8))
    plt.plot(history['train_acc'], 'b-', linewidth=2, label='Training Accuracy')
    plt.plot(history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy of AttBiLSTM with SHAP FS', fontsize=28, pad=20)
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Accuracy (%)', fontsize=24)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(False)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
    plt.close()



def plot_roc_curve(roc_data, num_classes, save_path):
    """Plot ROC curves"""
    plt.figure(figsize=(10, 8))

    if num_classes == 2:
        # Binary classification
        fpr, tpr, auc_score = roc_data
        plt.plot(fpr, tpr, 'b-', linewidth=3,
                 label=f'ROC Curve (AUC = {auc_score:.4f})')
    else:
        # Multiclass classification
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        for i, (fpr, tpr, auc_score) in enumerate(roc_data):
            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, color=color, linewidth=2,
                     label=f'Class {i} (AUC = {auc_score:.4f})')

    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.title('ROC Curves - AttBiLSTM with SHAP FS', fontsize=28, pad=20)
    plt.legend(loc="lower right", fontsize=18)
    plt.grid(True, alpha=0.3)

    # Enhance aesthetics
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
    plt.close()



def save_model_architecture(model, file_path):
    """Save model architecture as text file"""
    with open(file_path, 'w') as f:
        f.write("Advanced AttBiLSTM Model Architecture with SHAP Feature Selection\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Input Dimension: {model.input_dim}\n")
        f.write(f"Hidden Dimension: {model.hidden_dim}\n")
        f.write(f"Number of Classes: {model.num_classes}\n")
        f.write(f"Number of LSTM Layers: {model.num_layers}\n\n")

        f.write("Feature Selection:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Original Features: {len(feature_names)}\n")
        f.write(f"Selected Features: {model.input_dim}\n")
        f.write(f"Feature Selection Method: SHAP\n\n")

        f.write("Model Architecture:\n")
        f.write("-" * 30 + "\n")
        f.write(str(model))
        f.write("\n\n")

        f.write("Model Parameters:\n")
        f.write("-" * 20 + "\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")

        f.write("\nLayer-wise Parameter Count:\n")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    f.write(f"{name}: {params:,} parameters\n")



def run_complete_analysis():
    """Run the complete AttBiLSTM analysis pipeline with SHAP feature selection"""

    print("=" * 60)
    print("Starting Advanced AttBiLSTM Analysis Pipeline with SHAP FS")
    print("=" * 60)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(X_train, y_train_initial, X_val, y_val_initial, batch_size=32)

    # Initialize model
    model = AdvancedAttBiLSTM(
        input_dim=num_features,
        num_classes=num_classes,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Input features: {num_features} (selected by SHAP from {len(feature_names)} original features)")

    # Train model
    print("\nTraining model...")
    trained_model, history, training_time = train_model(
        model, train_loader, val_loader, epochs=100, patience=15
    )

    # Save model
    model_save_path = os.path.join(BASE_DIR, 'models', 'attbilstm_shap_model.pth')
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
            'mean_': scaler.mean_.tolist(),
            'scale_': scaler.scale_.tolist()
        },
        'selected_features': selected_features,
        'selected_indices': selected_indices.tolist(),
        'feature_importance_scores': feature_importance_scores.tolist()
    }, model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Save model architecture as text
    architecture_path = os.path.join(BASE_DIR, 'models', 'model_architecture.txt')
    save_model_architecture(trained_model, architecture_path)
    print(f"Model architecture saved to: {architecture_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    trained_model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    start_time = time.time()
    with torch.no_grad():
        test_outputs, attention_weights = trained_model(X_test_tensor)
        test_probabilities = F.softmax(test_outputs, dim=1).cpu().numpy()
        test_predictions = np.argmax(test_probabilities, axis=1)
    testing_time = time.time() - start_time

    # Calculate detailed metrics
    detailed_metrics, confusion_mat, roc_data = calculate_detailed_metrics(
        y_test_initial, test_predictions, test_probabilities
    )

    # Add timing and feature selection information
    detailed_metrics['Training Time'] = training_time
    detailed_metrics['Testing Time'] = testing_time
    detailed_metrics['Original Features'] = len(feature_names)
    detailed_metrics['Selected Features'] = num_features
    detailed_metrics['Feature Selection Method'] = 'SHAP'

    # Print results
    print("\n" + "=" * 50)
    print("DETAILED EVALUATION RESULTS")
    print("=" * 50)

    # Create results table
    results_df = pd.DataFrame([detailed_metrics])
    results_df = results_df.round(4)
    print("\nMain Metrics:")
    main_metrics = ['ACC', 'AUC', 'PRE', 'SP', 'SN', 'F1', 'MCC', 'Training Time', 'Testing Time']
    print(results_df[main_metrics].to_string(index=False))

    print("\nDetailed Classification Metrics:")
    detailed_classification = ['TPR', 'FPR', 'TNR', 'FNR', 'TP', 'TN', 'FP', 'FN']
    print(results_df[detailed_classification].to_string(index=False))

    print(f"\nFeature Selection Results:")
    print(f"Original Features: {detailed_metrics['Original Features']}")
    print(f"Selected Features: {detailed_metrics['Selected Features']}")
    print(
        f"Feature Reduction: {(1 - detailed_metrics['Selected Features'] / detailed_metrics['Original Features']) * 100:.2f}%")

    # Save results to CSV
    results_path = os.path.join(BASE_DIR, 'results', 'detailed_metrics.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(BASE_DIR, 'results', 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")

    # Save selected features
    selected_features_df = pd.DataFrame({
        'feature_name': selected_features,
        'original_index': selected_indices,
        'shap_importance': feature_importance_scores[selected_indices]
    })
    selected_features_path = os.path.join(BASE_DIR, 'feature_selection', 'selected_features.csv')
    selected_features_df.to_csv(selected_features_path, index=False)
    print(f"Selected features saved to: {selected_features_path}")

    # Generate and save plots
    print("\nGenerating plots...")

    # 1. Confusion Matrix
    class_names = [f'Class {i}' for i in range(num_classes)]
    cm_path = os.path.join(BASE_DIR, 'plots', 'confusion_matrix.png')
    plot_confusion_matrix(confusion_mat, class_names, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    # 2. Training Curves
    curves_path_loss_curve = os.path.join(BASE_DIR, 'plots', 'loss_curve.png')
    curves_path_accuracy_curve = os.path.join(BASE_DIR, 'plots', 'accuracy_curve.png')
    plot_training_loss(history, curves_path_loss_curve)
    plot_training_accuracy(history, curves_path_accuracy_curve)
    print(f"Training curves saved to: {curves_path_loss_curve, curves_path_accuracy_curve}")

    # 3. ROC Curves
    roc_path = os.path.join(BASE_DIR, 'plots', 'roc_curves.png')
    plot_roc_curve(roc_data, num_classes, roc_path)
    print(f"ROC curves saved to: {roc_path}")

    # Save detailed classification report
    class_report = classification_report(y_test_initial, test_predictions,
                                         target_names=class_names,
                                         output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_path = os.path.join(BASE_DIR, 'results', 'classification_report.csv')
    class_report_df.to_csv(class_report_path)
    print(f"Classification report saved to: {class_report_path}")

    # Save attention weights analysis (sample)
    if attention_weights:
        attention_analysis_path = os.path.join(BASE_DIR, 'results', 'attention_analysis.txt')
        with open(attention_analysis_path, 'w') as f:
            f.write("Attention Weights Analysis\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Number of attention heads: {len(attention_weights)}\n")
            f.write(f"Attention tensor shape: {attention_weights[0].shape}\n\n")

            # Calculate average attention across all samples and heads
            avg_attention = torch.mean(torch.stack(attention_weights), dim=0)
            f.write(f"Average attention statistics:\n")
            f.write(f"Mean: {torch.mean(avg_attention).item():.6f}\n")
            f.write(f"Std: {torch.std(avg_attention).item():.6f}\n")
            f.write(f"Min: {torch.min(avg_attention).item():.6f}\n")
            f.write(f"Max: {torch.max(avg_attention).item():.6f}\n")

        print(f"Attention analysis saved to: {attention_analysis_path}")

    # Create summary report
    summary_path = os.path.join(BASE_DIR, 'results', 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Advanced AttBiLSTM Analysis Summary with SHAP Feature Selection\n")
        f.write("=" * 65 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Original number of features: {len(feature_names)}\n")
        f.write(f"Selected number of features: {num_features}\n")
        f.write(f"Feature reduction ratio: {(1 - num_features / len(feature_names)) * 100:.2f}%\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")

        f.write("Feature Selection Information:\n")
        f.write(f"Method: SHAP (SHapley Additive exPlanations)\n")
        f.write(f"Base model for SHAP: Random Forest\n")
        f.write(f"Target features: 2000\n")
        f.write(f"Actually selected: {num_features}\n\n")

        f.write("Model Configuration:\n")
        f.write(f"Hidden dimension: 128\n")
        f.write(f"Number of LSTM layers: 2\n")
        f.write(f"Dropout rate: 0.3\n")
        f.write(f"Total parameters: {sum(p.numel() for p in trained_model.parameters()):,}\n\n")

        f.write("Performance Metrics:\n")
        for metric, value in detailed_metrics.items():
            if isinstance(value, (int, float)) and metric not in ['Original Features', 'Selected Features']:
                f.write(f"{metric}: {value:.4f}\n")

        f.write(f"\nClass distribution in test set:\n")
        for i, count in enumerate(np.bincount(y_test_initial)):
            f.write(f"Class {i}: {count} samples ({count / len(y_test_initial) * 100:.2f}%)\n")

    print(f"Analysis summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"All results saved in directory: {BASE_DIR}")
    print("\nFiles generated:")
    print("- Models: attbilstm_shap_model.pth, model_architecture.txt")
    print("- Feature Selection: shap_results.json, selected_features.csv, shap_feature_importance.png")
    print("- Results: detailed_metrics.csv, training_history.csv, classification_report.csv")
    print("- Plots: confusion_matrix.png/pdf, training_curves.png/pdf, roc_curves.png/pdf")
    print("- Analysis: attention_analysis.txt, analysis_summary.txt")

    return trained_model, detailed_metrics, history



if __name__ == "__main__":
    try:
        final_model, final_metrics, final_history = run_complete_analysis()

        # Print final performance summary
        print("\n" + "=" * 60)
        print("FINAL PERFORMANCE SUMMARY")
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
        print(
            f"Feature Reduction: {(1 - final_metrics['Selected Features'] / final_metrics['Original Features']) * 100:.2f}%")

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        print("\nMemory cleanup completed.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("Analysis pipeline finished.")
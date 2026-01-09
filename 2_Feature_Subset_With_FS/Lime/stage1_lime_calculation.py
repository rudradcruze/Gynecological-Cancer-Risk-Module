import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lime
import lime.lime_tabular
import warnings

warnings.filterwarnings('ignore')

# Check for MPS availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Input and output directories
input_directory = '../../dataset'  # Adjusted to the correct location for input datasets
output_root_directory = './Processed_Datasets'

# Create output subdirectories if they don't exist
score_dir = os.path.join(output_root_directory, 'Score')
os.makedirs(score_dir, exist_ok=True)


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_model(X_train, y_train, X_val, y_val, input_size, num_classes, epochs=100):
    """Train a PyTorch neural network model"""
    model = SimpleNN(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(y_val, val_predictions.cpu().numpy())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)
    return model


def pytorch_predict_proba(X):
    """Prediction function for LIME that works with PyTorch model"""
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()


def analyze_lime_explanations(explanations, feature_names, dataset_name, top_k=None):
    """Analyze and aggregate LIME explanations across multiple instances"""
    feature_importance_sum = {}
    feature_importance_count = {}

    for exp in explanations:
        for feature_idx, importance in exp.as_list():
            # Extract feature name from the explanation string
            feature_name = feature_idx.split('<=')[0].split('>')[0].strip()

            if feature_name in feature_importance_sum:
                feature_importance_sum[feature_name] += abs(importance)
                feature_importance_count[feature_name] += 1
            else:
                feature_importance_sum[feature_name] = abs(importance)
                feature_importance_count[feature_name] = 1

    # Calculate average importance for all features
    feature_avg_importance = {}
    for feature in feature_names:
        if feature in feature_importance_sum:
            feature_avg_importance[feature] = feature_importance_sum[feature] / feature_importance_count[feature]
        else:
            feature_avg_importance[feature] = 0.0

    # Sort by importance and return all features with their scores
    sorted_features = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)

    return sorted_features


# Process each dataset file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        dataset_path = os.path.join(input_directory, filename)
        dataset_name = os.path.splitext(filename)[0]

        try:
            # Load the dataset
            df = pd.read_csv(dataset_path)
            print(f"\nProcessing dataset: {dataset_name}")
            print(f"Dataset shape: {df.shape}")
        except Exception as e:
            print(f"Error reading file '{filename}': {e}. Skipping this file.")
            continue

        # Ensure the dataset has enough columns
        if df.shape[1] < 2:
            print(f"File '{filename}' does not have enough columns. Skipping this file.")
            continue

        try:
            # Split features and target (last column is always target)
            X = df.iloc[:, :-1].copy()  # All columns except last (features)
            y = df.iloc[:, -1].copy()  # Last column (target)

            print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

            # Handle categorical target variable
            label_encoder = LabelEncoder()
            if y.dtype == 'object' or y.dtype.name == 'category':
                y_encoded = label_encoder.fit_transform(y)
                class_names = label_encoder.classes_.tolist()
            else:
                y_encoded = y.astype(int).values
                class_names = [str(i) for i in sorted(np.unique(y_encoded))]

            print(f"Classes: {class_names}")

            # Ensure features are numeric - genomic data should already be numeric
            # But handle any potential issues
            if X.dtypes.apply(lambda x: x == 'object').any():
                print("Converting non-numeric features to numeric...")
                X = X.apply(pd.to_numeric, errors='coerce')
                X = X.fillna(0)  # Fill NaN with 0 for genomic data

            # Store original feature names
            feature_names = X.columns.tolist()

            # Convert to numpy arrays
            X_array = X.values.astype(np.float32)
            y_array = y_encoded.astype(np.int64)

            # Remove constant features (genes with no variation)
            feature_variance = np.var(X_array, axis=0)
            non_constant_mask = feature_variance > 1e-8  # Very small threshold for genomic data

            if np.sum(non_constant_mask) < X_array.shape[1]:
                removed_count = X_array.shape[1] - np.sum(non_constant_mask)
                print(f"Removing {removed_count} constant/near-constant features")
                X_array = X_array[:, non_constant_mask]
                feature_names = [feature_names[i] for i in range(len(feature_names)) if non_constant_mask[i]]

            if X_array.shape[1] == 0:
                print(f"No valid features remaining for '{filename}'. Skipping.")
                continue

            print(f"Final feature count: {X_array.shape[1]}")

            # Scale features (important for genomic data)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)

            # Split the data with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_array, test_size=0.2, random_state=42, stratify=y_array
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

            print(f"Running LIME feature analysis for '{dataset_name}'...")

            # Train PyTorch model
            num_classes = len(np.unique(y_array))
            input_size = X_train.shape[1]

            print(f"Training neural network model...")

            model = train_model(X_train, y_train, X_val, y_val, input_size, num_classes)

            # Test model performance
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                test_outputs = model(X_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1)
                test_accuracy = accuracy_score(y_test, test_predictions.cpu().numpy())

            print(f"Test Accuracy: {test_accuracy:.4f}")

            # Initialize LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification'
            )

            # Generate LIME explanations for a sample of test instances
            sample_size = min(20, len(X_test))  # Reduce sample size for large genomic datasets
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)

            explanations = []

            for i, idx in enumerate(sample_indices):
                if i % 5 == 0:
                    print(f"Processing explanation {i + 1}/{sample_size}")

                exp = explainer.explain_instance(
                    X_test[idx],
                    pytorch_predict_proba,
                    num_features=len(feature_names),  # Get scores for all features
                    num_samples=500  # Reduce samples for speed
                )
                explanations.append(exp)

            # Analyze aggregated feature importance for ALL features
            all_feature_scores = analyze_lime_explanations(explanations, feature_names, dataset_name)

            # Create a dictionary for easy lookup of scores by feature name
            score_dict = dict(all_feature_scores)

            # Extract scores in ORIGINAL feature order (not sorted)
            lime_scores = [score_dict[feature] for feature in feature_names]

            # Normalize the LIME scores to a range of 0 to 1 (similar to Boruta)
            min_score, max_score = np.min(lime_scores), np.max(lime_scores)
            if max_score > min_score:
                normalized_scores = (np.array(lime_scores) - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros(len(lime_scores))

            # Create ranking (1 = best, higher number = lower rank) - based on scores but keeping original order
            rankings = np.argsort(np.argsort(lime_scores)[::-1]) + 1

            # Determine "selected" features (top 25% or features above median)
            threshold = np.percentile(lime_scores, 75)  # Top 25%
            selected_features = [score >= threshold for score in lime_scores]

            # Create a DataFrame to hold features and their scores (similar to Boruta output)
            # Features are kept in ORIGINAL order, not sorted by importance
            feature_results = pd.DataFrame({
                'Feature': feature_names,  # Original order maintained
                'LIME_Score': lime_scores,  # Scores in original feature order
                'Normalized_Score': normalized_scores,
                'Selected': selected_features,
                'Ranking': rankings
            })

            # Save the LIME scores to a CSV file in the Score directory
            lime_output_file = os.path.join(score_dir, f'{dataset_name}_LIME_Scores.csv')
            feature_results.to_csv(lime_output_file, index=False)

            print(f"Features and their LIME Scores for '{dataset_name}' saved to '{lime_output_file}'")

            # Print summary statistics
            confirmed_count = sum(selected_features)
            total_features = len(selected_features)

            print(f"Summary for '{dataset_name}':")
            print(f"  - Confirmed features: {confirmed_count}")
            print(f"  - Total features: {total_features}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing file '{filename}': {e}. Skipping this file.")
            continue

print("All datasets have been processed with LIME feature analysis.")
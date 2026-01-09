import os
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Input and output directories
input_directory = '../../dataset'  # Adjusted to the correct location for input datasets
output_root_directory = './Processed_Datasets'

# Create output subdirectories if they don't exist
score_dir = os.path.join(output_root_directory, 'Score')
os.makedirs(score_dir, exist_ok=True)

# Process each dataset file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        dataset_path = os.path.join(input_directory, filename)
        dataset_name = os.path.splitext(filename)[0]  # Get the dataset name without extension

        try:
            # Load the dataset
            df = pd.read_csv(dataset_path)
        except Exception as e:
            print(f"Error reading file '{filename}': {e}. Skipping this file.")
            continue

        # Ensure the dataset has enough columns to separate features and target
        if df.shape[1] < 2:
            print(f"File '{filename}' does not have enough columns. Skipping this file.")
            continue

        try:
            # Assuming the last column is the target column
            X = df.iloc[:, :-1]  # All columns except the last one (features)
            y = df.iloc[:, -1]  # Last column (target)

            # Handle categorical target variable
            label_encoder = LabelEncoder()
            if y.dtype == 'object' or y.dtype.name == 'category':
                y = label_encoder.fit_transform(y)

            # Ensure that feature data is numeric
            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.fillna(0)  # Fill NaN values resulting from non-numeric data

            # Convert to numpy arrays for Boruta
            X_array = X.values
            y_array = y.values

            # Initialize Random Forest classifier for Boruta
            rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)

            # Initialize Boruta feature selector
            feat_selector = BorutaPy(
                rf,
                n_estimators='auto',
                verbose=2,
                random_state=42,
                max_iter=100  # Maximum number of iterations
            )

            print(f"Running Boruta feature selection for '{dataset_name}'...")

            # Fit Boruta
            feat_selector.fit(X_array, y_array)

            # Get feature rankings and scores
            selected_features = feat_selector.support_
            tentative_features = feat_selector.support_weak_
            feature_rankings = feat_selector.ranking_

            # Get feature importance scores from Boruta's internal scores
            try:
                boruta_scores = feat_selector.imp_real_mean
            except AttributeError:
                # Fallback: fit a new Random Forest to get importance scores
                rf_temp = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
                rf_temp.fit(X_array, y_array)
                boruta_scores = rf_temp.feature_importances_

            # Normalize the Boruta scores to a range of 0 to 1 (similar to ANOVA F-test)
            min_score, max_score = np.min(boruta_scores), np.max(boruta_scores)
            normalized_scores = (boruta_scores - min_score) / (
                        max_score - min_score + 1e-10)  # Add small constant to avoid division by zero

            # Create a DataFrame to hold features and their scores (similar to ANOVA output)
            feature_results = pd.DataFrame({
                'Feature': X.columns,
                'Boruta_Score': boruta_scores,
                'Normalized_Score': normalized_scores,
                'Selected': selected_features,
                'Ranking': feature_rankings
            })

            # Save the Boruta scores to a CSV file in the Score directory
            boruta_output_file = os.path.join(score_dir, f'{dataset_name}_Boruta_Scores.csv')
            feature_results.to_csv(boruta_output_file, index=False)

            print(f"Features and their Boruta Scores for '{dataset_name}' saved to '{boruta_output_file}'")

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

print("All datasets have been processed with Boruta feature selection.")
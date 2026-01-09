import os
import pandas as pd

# Input and output directories
input_directory = '../../dataset'
output_root_directory = './Processed_Datasets'
score_dir = os.path.join(output_root_directory, 'Score')
report_dir = os.path.join(output_root_directory, 'Report')
selection_dir = os.path.join(output_root_directory, 'Selection')

# Ensure output directories exist
os.makedirs(report_dir, exist_ok=True)
os.makedirs(selection_dir, exist_ok=True)

# List all Boruta files to display options to the user
datasets = [f for f in os.listdir(score_dir) if f.endswith('_Boruta_Scores.csv') and not f.startswith('._')]
# datasets = [f for f in os.listdir(score_dir)]
dataset_names = [os.path.splitext(f)[0].replace('_Boruta_Scores', '') for f in datasets]

# Display options to the user
print("Available datasets:")
for idx, name in enumerate(dataset_names):
    print(f"{idx + 1}. {name}")

# Ask user to select datasets to process (e.g., "1 3 5")
selected_indices = input("\nEnter the indices of the datasets to process (e.g., 1 3): ").strip().split()
selected_indices = [int(i) - 1 for i in selected_indices if i.isdigit() and 0 <= int(i) - 1 < len(datasets)]

# Process each selected dataset
for idx in selected_indices:
    dataset_name = dataset_names[idx]
    score_file = os.path.join(score_dir, datasets[idx])

    # Load the Boruta's
    feature_scores = pd.read_csv(score_file)
    
    # Load the original dataset to retrieve features and target
    dataset_file = os.path.join(input_directory, f"{dataset_name}.csv")
    df = pd.read_csv(dataset_file)
    X = df.iloc[:, :-1]  # All columns except the last one (features)
    y = df.iloc[:, -1]   # Last column (target)

    # column number
    column_number = 1

    # Sort the features by Boruta in descending order
    sorted_feature_scores = feature_scores.sort_values(by='Boruta_Score', ascending=False)

    # Ask user for selection method for this dataset
    print(f"\nSelect an option for {dataset_name}:")
    print("1. Select features by score threshold")
    print("2. Select top N features")

    choice = int(input("\nEnter your choice (1 or 2): "))
    report_details = {"dataset_name": dataset_name, "chosen_option": "", "selected_feature_count": 0}
    type = ''
    value = 0

    if choice == 1:
        # Option 1: Select features by score threshold
        threshold = float(input("Enter the score threshold: "))
        selected_features = sorted_feature_scores[sorted_feature_scores['Boruta_Score'] >= threshold]['Feature']
        report_details["chosen_option"] = f"Score threshold >= {threshold}"
        report_details["selected_feature_count"] = len(selected_features)
        type = 'Threshold'
        value = threshold

    elif choice == 2:
        # Option 2: Select top N features
        top_n = int(input("Enter the number of top features to select: "))
        selected_features = sorted_feature_scores.head(top_n)['Feature']
        report_details["chosen_option"] = f"Top {top_n} features"
        report_details["selected_feature_count"] = top_n
        type = 'Top'
        value = top_n

    else:
        print("Invalid choice. Skipping this dataset.")
        continue

    # Select features and concatenate with the target
    X_selected = X[selected_features]
    X_selected_with_target = pd.concat([X_selected, y], axis=1)

    # Save the selected features along with the target to a CSV file in the Selection directory
    selection_output_file = os.path.join(selection_dir, f'{dataset_name}_Selected_{type}_{value}_Features.csv')
    X_selected_with_target.to_csv(selection_output_file, index=False)
    print(f"Selected features and target column saved to '{selection_output_file}'")

    # Save the report details to a text file in the report directory
    report_file = os.path.join(report_dir, f'{dataset_name}_Feature_Selection_Report.txt')
    with open(report_file, 'w') as f:
        f.write("Boruta Feature Selection Report\n")
        f.write("=====================================\n")
        f.write(f"Dataset Name: {dataset_name}\n")
        f.write(f"Selected Option: {report_details['chosen_option']}\n")
        f.write(f"Number of Selected Features: {report_details['selected_feature_count']}\n")
        f.write(f"Output CSV File for Selected Features: {selection_output_file}\n")
        f.write(f"Borutas CSV File: {score_file}\n")

    print(f"Report for '{dataset_name}' saved to '{report_file}'\n")

print("All selected datasets have been processed successfully.")
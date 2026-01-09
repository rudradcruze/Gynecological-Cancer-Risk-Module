import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import numpy as np

def plot_dynamic_subplots_from_folder(folder_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all CSV files in the folder
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv') and not f.startswith('._')]
    num_files = len(csv_files)
    num_rows = 2  # Two rows: one for line plot, one for scatter plot
    num_cols = num_files  # Each column represents one dataset

    # Define figure size with each dataset in a separate column
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_files * 4, 8))

    # Ensure axs is treated as a 2D array for consistency
    if num_files == 1:
        axs = np.array([[axs[0]], [axs[1]]])
    else:
        axs = np.atleast_2d(axs)

    for i, csv_file in enumerate(csv_files):
        try:
            # Extract the title from the filename
            file_name = os.path.basename(csv_file)
            title_part = file_name.split('_')[0]  # Take the part before the first underscore

            # Load the CSV file
            df = pd.read_csv(csv_file)
            # Visualizing Column Number
            column_number = 1
            df_sorted = df.sort_values(by=df.columns[column_number], ascending=False)

            # Plot the sorted data as a curve plot in the top row
            axs[0, i].set_facecolor('lightgray')
            axs[0, i].plot(range(len(df_sorted)), df_sorted[df_sorted.columns[column_number]], color='orange', linewidth=2)
            axs[0, i].set_title(f'{title_part} (A)', fontsize=10)
            axs[0, i].set_ylabel('Boruta', fontsize=8)
            axs[0, i].set_xlabel('Number of gene', fontsize=8)
            
            # Set custom ticks for x and y axes for less clutter
            axs[0, i].xaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce x-axis ticks
            axs[0, i].yaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce y-axis ticks
            axs[0, i].grid(True, linestyle='-', color='white', linewidth=0.6)

            # Plot the raw data as a scatter plot in the bottom row
            axs[1, i].set_facecolor('lightgray')
            axs[1, i].scatter(range(len(df)), df[df.columns[column_number]], color='#6594e0', s=20)  # Reduced marker size
            axs[1, i].set_title(f'{title_part} (B)', fontsize=10)
            axs[1, i].set_ylabel('Boruta', fontsize=8)
            axs[1, i].set_xlabel('Number of gene', fontsize=8)
            
            # Set custom ticks for x and y axes
            axs[1, i].xaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce x-axis ticks
            axs[1, i].yaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce y-axis ticks
            axs[1, i].grid(True, linestyle='-', color='white', linewidth=0.6)
        
        except Exception as e:
            print(f"Error processing file '{csv_file}': {e}. Skipping this file.")
            continue

    # Optimize layout spacing
    plt.tight_layout()  # Automatically adjust subplot parameters for a tight layout

    # Save the figure as an image file in the specified output folder
    output_path_base = os.path.join(output_folder, f'combined_plot_boruta_column_{column_number}')
    plt.savefig(f'{output_path_base}.png', format='png', dpi=1000)
    plt.savefig(f'{output_path_base}.jpg', format='jpg', dpi=1000)
    plt.savefig(f'{output_path_base}.pdf', format='pdf', dpi=1000)
    plt.savefig(f'{output_path_base}.svg', format='svg', dpi=1000)
    
    # Show the plot
    # plt.show()

# Example usage
folder_path = './Processed_Datasets/Score'  # Folder containing the CSV files
output_folder = './Processed_Datasets/Plots'  # Folder to save the output plot
plot_dynamic_subplots_from_folder(folder_path, output_folder)
print("Plotting completed successfully.")
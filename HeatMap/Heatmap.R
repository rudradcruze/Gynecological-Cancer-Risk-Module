# Install necessary packages if not already installed
if (!requireNamespace("magrittr", quietly = TRUE)) {
  install.packages("magrittr")
}
if (!requireNamespace("pheatmap", quietly = TRUE)) {
  install.packages("pheatmap")
}

# Load required libraries
library(pheatmap)
library(matrixStats)
library(magrittr)

# Read the dataset and set the first column (genes) as row names
df <- read.csv("output_files/matched_filtered_data_GSE33630.csv", header = TRUE)

# Make gene names unique
df$genes <- make.unique(as.character(df$genes))

# Set the 'genes' column as row names and then remove it from the data
rownames(df) <- df$genes
df <- df[, -1]  # Remove the 'genes' column from the dataframe

# Convert the dataset into a matrix for the heatmap
Data1 <- as.matrix(df)

# Create the heatmap
pheatmap(
  Data1, 
  show_rownames = TRUE,      # Show row names on the left
  show_colnames = TRUE, 
  fontsize_row = 8,          # Adjust font size if needed
  fontsize_col = 10          # Adjust font size if needed
)

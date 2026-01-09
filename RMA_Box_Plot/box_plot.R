# Install necessary packages if you haven't
if (!requireNamespace("affy", quietly = TRUE)) {
  install.packages("BiocManager")
  BiocManager::install("affy")
}

if (!requireNamespace("affyPLM", quietly = TRUE)) {
  BiocManager::install("affyPLM")
}

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}

library(affy)
library(affyPLM)
update.packages(ask = FALSE, checkBuilt = TRUE)

# Set the directory where your CEL files are located
cel_files_directory <- "GSE120490"  # Replace with your actual path

# Read the CEL files
data_raw <- ReadAffy(celfile.path = cel_files_directory)

# Get sample names to use as x-axis labels
sample_names <- sampleNames(data_raw)
par(mar = c(8, 4, 4, 2) + 0.1)  # Increase the bottom margin (first number)

# Define a color palette (adjust as needed)
color_palette <- rainbow(length(sample_names))  # Generates different colors for each box

# Create the box plot with different colors for each box
boxplot(data_raw, 
        main = "Original", 
        col = color_palette,  # Use the color palette for different boxes
        names = sample_names, 
        las = 2,              # Vertical x-axis labels
        whisklty = 1,         # Solid line for lower whisker
        staplelty = 1,        # Solid line for the staple (end of whiskers)
        boxlty = 1,           # Solid line for the box border
        medlty = 1,           # Solid line for the median
        cex.axis = 0.8)       # Reduce the font size of x-axis labels     # Solid line for the median

# Normalize the data using RMA
data_normalized <- rma(data_raw)

# Summary before normalization
boxplot(data_normalized, 
        main = "RMA", 
        col = color_palette,  # Use the color palette for different boxes
        names = sample_names, 
        las = 2,              # Vertical x-axis labels
        whisklty = 1,         # Solid line for lower whisker
        staplelty = 1,        # Solid line for the staple (end of whiskers)
        boxlty = 1,           # Solid line for the box border
        medlty = 1,           # Solid line for the median
        cex.axis = 0.8)       # Reduce the font size of x-axis labels     # Solid line for the median



# Extract expression values and convert to a data frame
expression_matrix <- exprs(data_normalized)

# Optionally, convert to a data frame and set row names
expression_df <- as.data.frame(expression_matrix)
rownames(expression_df) <- featureNames(data_normalized)

# View the first few rows of the expression data
head(expression_df)

# Save the expression data as a CSV file
write.csv(expression_df, file = "GSE5281_series_matrix.csv", row.names = TRUE)


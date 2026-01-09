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

# CONFIGURATION
MAX_SAMPLES <- 50  # Change this number to limit columns
GSE_NUMBER <- "GSE120490"  # Dataset identifier
HIGH_DPI <- 300  # High resolution for images

# Create main output directory
output_dir <- paste0(GSE_NUMBER, "_heatmap_analysis")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created directory:", output_dir, "\n")
}

# Read the dataset
df <- read.csv("output_files/matched_filtered_data_GSE120490.csv", header = TRUE)

# DEBUG: Check the structure of your data
cat("=== DEBUGGING DATA STRUCTURE ===\n")
cat("Dataset dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
cat("Column names:", paste(colnames(df)[1:min(10, ncol(df))], collapse = ", "))
if (ncol(df) > 10) cat(", ...")
cat("\n")

# Check for Gene.symbol column specifically
if ("Gene.symbol" %in% colnames(df)) {
  cat("Found 'Gene.symbol' column - using as gene names\n")
  
  # Make gene names unique and clean
  gene_names <- as.character(df$Gene.symbol)
  gene_names[is.na(gene_names) | gene_names == ""] <- paste0("Unknown_Gene_", seq_along(gene_names[is.na(gene_names) | gene_names == ""]))
  gene_names <- make.unique(gene_names)
  
  # Set gene names as row names and remove the Gene.symbol column
  rownames(df) <- gene_names
  df <- df[, -which(colnames(df) == "Gene.symbol")]
  
} else if ("genes" %in% colnames(df)) {
  cat("Found 'genes' column - using as gene names\n")
  
  # Make gene names unique
  df$genes <- make.unique(as.character(df$genes))
  rownames(df) <- df$genes
  df <- df[, -which(colnames(df) == "genes")]
  
} else {
  cat("No 'Gene.symbol' or 'genes' column found. Checking first column...\n")
  
  # Check if first column contains gene names (non-numeric)
  first_col <- df[, 1]
  if (is.character(first_col) || is.factor(first_col)) {
    cat("Using first column as gene names\n")
    gene_names <- make.unique(as.character(first_col))
    rownames(df) <- gene_names
    df <- df[, -1]  # Remove first column
  } else {
    cat("Creating generic gene names\n")
    rownames(df) <- paste0("Gene_", 1:nrow(df))
  }
}

# Print dataset information
cat("Original dataset dimensions:", nrow(df), "genes x", ncol(df), "samples\n")
cat("Gene names preview:", paste(head(rownames(df), 5), collapse = ", "), "...\n")

# Randomly sample N columns (reproducible with set.seed)
set.seed(123)  # For reproducible results
if (ncol(df) > MAX_SAMPLES) {
  selected_cols <- sample(1:ncol(df), MAX_SAMPLES)
  selected_cols <- sort(selected_cols)  # Keep original order
  df_limited <- df[, selected_cols]
  cat("Randomly selected", MAX_SAMPLES, "samples from", ncol(df), "total samples\n")
  cat("Selected sample indices:", paste(selected_cols[1:min(10, length(selected_cols))], collapse = ", "))
  if (length(selected_cols) > 10) cat(", ...")
  cat("\n")
} else {
  df_limited <- df
  cat("Using all", ncol(df), "samples (less than maximum of", MAX_SAMPLES, ")\n")
}

# Convert the limited dataset into a matrix for the heatmap
Data1 <- as.matrix(df_limited)
cat("Final heatmap dimensions:", nrow(Data1), "genes x", ncol(Data1), "samples\n")

# Function to save heatmap in multiple formats - WITH Times New Roman font and TIGHT LAYOUT
save_heatmap_multiple_formats <- function(data, base_filename, title, output_directory = output_dir, 
                                          fontsize_row = 10, fontsize_col = 12, 
                                          width = NULL, height = NULL) {
  
  # File paths for different formats
  png_file <- file.path(output_directory, paste0(base_filename, ".png"))
  jpg_file <- file.path(output_directory, paste0(base_filename, ".jpg"))
  pdf_file <- file.path(output_directory, paste0(base_filename, ".pdf"))
  
  cat("Saving heatmap in multiple formats with Times New Roman font and tight layout...\n")
  
  # Calculate optimal dimensions based on data size
  if (is.null(width)) {
    width <- max(8, min(20, ncol(data) * 0.3 + 4))  # Dynamic width based on samples
  }
  if (is.null(height)) {
    height <- max(6, min(16, nrow(data) * 0.4 + 3))  # Dynamic height based on genes
  }
  
  cat("Using dimensions:", width, "x", height, "inches\n")
  
  # Set font family to Times New Roman
  par(family = "serif")  # serif = Times New Roman family
  
  # PNG format (high DPI) - WITH Times New Roman and TIGHT LAYOUT
  png(png_file, width = width, height = height, units = "in", res = HIGH_DPI, family = "serif")
  pheatmap(
    data,
    show_rownames = TRUE,      # Show gene names
    show_colnames = TRUE,      # Show sample names
    fontsize_row = fontsize_row,
    fontsize_col = fontsize_col,
    main = title,
    clustering_distance_rows = "euclidean",
    clustering_distance_cols = "euclidean",
    clustering_method = "complete",
    border_color = "white",    # Add borders for better separation
    cellwidth = 20,           # Optimal cell width
    cellheight = 20,          # Optimal cell height
    fontfamily = "serif",     # Times New Roman font family
    # TIGHT LAYOUT SETTINGS
    margin = c(1, 1),         # Minimal margins [bottom, right]
    treeheight_row = 50,      # Smaller dendrogram height
    treeheight_col = 50       # Smaller dendrogram height
  )
  dev.off()
  cat("✓ Saved PNG with Times New Roman (tight layout):", png_file, "\n")
  
  # JPEG format (high DPI, high quality) - WITH Times New Roman and TIGHT LAYOUT
  jpeg(jpg_file, width = width, height = height, units = "in", res = HIGH_DPI, quality = 95, family = "serif")
  pheatmap(
    data,
    show_rownames = TRUE,      # Show gene names
    show_colnames = TRUE,      # Show sample names
    fontsize_row = fontsize_row,
    fontsize_col = fontsize_col,
    main = title,
    clustering_distance_rows = "euclidean",
    clustering_distance_cols = "euclidean",
    clustering_method = "complete",
    border_color = "white",    # Add borders for better separation
    cellwidth = 20,           # Optimal cell width
    cellheight = 20,          # Optimal cell height
    fontfamily = "serif",     # Times New Roman font family
    # TIGHT LAYOUT SETTINGS
    margin = c(1, 1),         # Minimal margins [bottom, right]
    treeheight_row = 50,      # Smaller dendrogram height
    treeheight_col = 50       # Smaller dendrogram height
  )
  dev.off()
  cat("✓ Saved JPEG with Times New Roman (tight layout):", jpg_file, "\n")
  
  # PDF format (vector, scalable) - WITH Times New Roman and TIGHT LAYOUT
  pdf(pdf_file, width = width, height = height, family = "serif")
  pheatmap(
    data,
    show_rownames = TRUE,      # Show gene names
    show_colnames = TRUE,      # Show sample names
    fontsize_row = fontsize_row,
    fontsize_col = fontsize_col,
    main = title,
    clustering_distance_rows = "euclidean",
    clustering_distance_cols = "euclidean",
    clustering_method = "complete",
    border_color = "white",    # Add borders for better separation
    cellwidth = 20,           # Optimal cell width
    cellheight = 20,          # Optimal cell height
    fontfamily = "serif",     # Times New Roman font family
    # TIGHT LAYOUT SETTINGS
    margin = c(1, 1),         # Minimal margins [bottom, right]
    treeheight_row = 50,      # Smaller dendrogram height
    treeheight_col = 50       # Smaller dendrogram height
  )
  dev.off()
  cat("✓ Saved PDF with Times New Roman (tight layout):", pdf_file, "\n")
  
  # Reset font family
  par(family = "")
  
  return(list(png = png_file, jpg = jpg_file, pdf = pdf_file))
}

# Create the main heatmap with gene names visible
main_title <- paste("Gene Expression Heatmap -", GSE_NUMBER, "\n(", nrow(Data1), "genes x", ncol(Data1), "samples)")

# Adjust font sizes based on number of genes and samples
fontsize_row <- ifelse(nrow(Data1) > 100, 8, ifelse(nrow(Data1) > 50, 10, 12))
fontsize_col <- ifelse(ncol(Data1) > 30, 8, ifelse(ncol(Data1) > 20, 10, 12))

cat("Using font sizes - Gene names:", fontsize_row, ", Sample names:", fontsize_col, "\n")

# Display heatmap in R console/viewer - WITH Times New Roman font and TIGHT LAYOUT
cat("Displaying heatmap with Times New Roman font, gene names, and tight layout...\n")

# Set font family to Times New Roman for display
par(family = "serif")

pheatmap(
  Data1, 
  show_rownames = TRUE,      # Show gene names on the left
  show_colnames = TRUE,      # Show sample names on top
  fontsize_row = fontsize_row,
  fontsize_col = fontsize_col,
  main = main_title,
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  clustering_method = "complete",
  border_color = "white",    # Add white borders for better separation
  cellwidth = 20,           # Optimal cell width
  cellheight = 20,          # Optimal cell height
  fontfamily = "serif",     # Times New Roman font family
  # TIGHT LAYOUT SETTINGS for display
  margin = c(1, 1),         # Minimal margins [bottom, right]
  treeheight_row = 50,      # Smaller dendrogram height
  treeheight_col = 50       # Smaller dendrogram height
)

# Save main heatmap in all formats
main_files <- save_heatmap_multiple_formats(
  Data1, 
  paste0(GSE_NUMBER, "_main_heatmap"), 
  main_title,
  fontsize_row = fontsize_row,
  fontsize_col = fontsize_col
)

# ENHANCED FUNCTION: Create multiple heatmaps with different sample subsets
create_subset_heatmaps <- function(data, samples_per_plot = 20, gse_id = GSE_NUMBER, 
                                   output_directory = output_dir) {
  
  # Create subdirectory for subset heatmaps
  subset_dir <- file.path(output_directory, "subset_heatmaps")
  if (!dir.exists(subset_dir)) {
    dir.create(subset_dir, recursive = TRUE)
  }
  
  total_samples <- ncol(data)
  num_plots <- ceiling(total_samples / samples_per_plot)
  
  cat("\nCreating", num_plots, "subset heatmaps with", samples_per_plot, "samples each\n")
  
  # Adjust font sizes for subsets
  subset_fontsize_row <- ifelse(nrow(data) > 100, 9, ifelse(nrow(data) > 50, 11, 13))
  subset_fontsize_col <- ifelse(samples_per_plot > 15, 9, 11)
  
  for (i in 1:num_plots) {
    start_col <- (i - 1) * samples_per_plot + 1
    end_col <- min(i * samples_per_plot, total_samples)
    
    subset_data <- data[, start_col:end_col, drop = FALSE]
    
    # Create title for subset
    subset_title <- paste("Gene Expression Subset", i, "-", gse_id, 
                          "\nSamples", start_col, "to", end_col,
                          "(", nrow(subset_data), "genes)")
    
    # Base filename for this subset
    base_filename <- paste0(gse_id, "_subset_", i, "_samples_", start_col, "_to_", end_col)
    
    # Save in all formats with tight layout
    subset_files <- save_heatmap_multiple_formats(
      subset_data,
      base_filename,
      subset_title,
      output_directory = subset_dir,
      fontsize_row = subset_fontsize_row,
      fontsize_col = subset_fontsize_col,
      width = NULL,  # Let function calculate optimal width
      height = NULL  # Let function calculate optimal height
    )
    
    cat("Subset", i, "completed\n")
  }
  
  cat("All subset heatmaps saved in:", subset_dir, "\n")
}

# Create subset heatmaps with gene names
create_subset_heatmaps(as.matrix(df), samples_per_plot = 25)

# BONUS: Create a summary report
create_summary_report <- function(original_data, limited_data, gse_id = GSE_NUMBER, 
                                  output_directory = output_dir) {
  
  report_file <- file.path(output_directory, paste0(gse_id, "_analysis_summary.txt"))
  
  # Prepare summary information
  summary_text <- paste(
    "=== GENE EXPRESSION HEATMAP ANALYSIS SUMMARY ===",
    paste("Dataset:", gse_id),
    paste("Analysis Date:", Sys.Date()),
    paste("Analysis Time:", Sys.time()),
    "",
    "=== DATA INFORMATION ===",
    paste("Original dimensions:", nrow(original_data), "genes x", ncol(original_data), "samples"),
    paste("Limited dimensions:", nrow(limited_data), "genes x", ncol(limited_data), "samples"),
    paste("Sample selection method: Random sampling with seed 123"),
    "",
    "=== GENE INFORMATION ===",
    paste("Total genes analyzed:", nrow(limited_data)),
    paste("First 10 genes:", paste(head(rownames(limited_data), 10), collapse = ", ")),
    if(nrow(limited_data) > 10) paste("Last 5 genes:", paste(tail(rownames(limited_data), 5), collapse = ", ")) else "",
    "",
    "=== SAMPLE INFORMATION ===",
    paste("Total samples visualized:", ncol(limited_data)),
    paste("Sample names:", paste(head(colnames(limited_data), 10), collapse = ", ")),
    if(ncol(limited_data) > 10) "..." else "",
    "",
    "=== FILES GENERATED ===",
    "Main heatmap files:",
    paste("- PNG:", paste0(gse_id, "_main_heatmap.png")),
    paste("- JPEG:", paste0(gse_id, "_main_heatmap.jpg")),
    paste("- PDF:", paste0(gse_id, "_main_heatmap.pdf")),
    "",
    "Subset heatmaps: Located in 'subset_heatmaps' subdirectory",
    "",
    "=== ANALYSIS PARAMETERS ===",
    paste("Maximum samples per plot:", MAX_SAMPLES),
    paste("Image resolution (DPI):", HIGH_DPI),
    paste("Clustering method: Complete linkage"),
    paste("Distance metric: Euclidean"),
    "",
    sep = "\n"
  )
  
  # Write summary to file
  writeLines(summary_text, report_file)
  cat("Summary report saved:", report_file, "\n")
}

# Generate summary report
create_summary_report(df, Data1)

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("All files saved in directory:", output_dir, "\n")
cat("✓ Main heatmap (PNG, JPEG, PDF)\n")
cat("✓ Subset heatmaps in 'subset_heatmaps' folder\n")
cat("✓ Analysis summary report\n")
cat("✓ Gene names are displayed on all heatmaps\n")

# Display final file structure
cat("\nGenerated file structure:\n")
if (file.exists(output_dir)) {
  files <- list.files(output_dir, recursive = TRUE, full.names = FALSE)
  cat(paste(files, collapse = "\n"))
} else {
  cat("Output directory not found")
}


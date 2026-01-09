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

# Create main output directory with "no_gene_names" suffix
output_dir <- paste0(GSE_NUMBER, "_heatmap_analysis_no_gene_names")
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
  cat("Found 'Gene.symbol' column - using as gene names (but not displaying)\n")
  
  # Make gene names unique and clean (for internal use only)
  gene_names <- as.character(df$Gene.symbol)
  gene_names[is.na(gene_names) | gene_names == ""] <- paste0("Unknown_Gene_", seq_along(gene_names[is.na(gene_names) | gene_names == ""]))
  gene_names <- make.unique(gene_names)
  
  # Set gene names as row names and remove the Gene.symbol column
  rownames(df) <- gene_names
  df <- df[, -which(colnames(df) == "Gene.symbol")]
  
} else if ("genes" %in% colnames(df)) {
  cat("Found 'genes' column - using as gene names (but not displaying)\n")
  
  # Make gene names unique
  df$genes <- make.unique(as.character(df$genes))
  rownames(df) <- df$genes
  df <- df[, -which(colnames(df) == "genes")]
  
} else {
  cat("No 'Gene.symbol' or 'genes' column found. Checking first column...\n")
  
  # Check if first column contains gene names (non-numeric)
  first_col <- df[, 1]
  if (is.character(first_col) || is.factor(first_col)) {
    cat("Using first column as gene names (but not displaying)\n")
    gene_names <- make.unique(as.character(first_col))
    rownames(df) <- gene_names
    df <- df[, -1]  # Remove first column
  } else {
    cat("Creating generic gene names (but not displaying)\n")
    rownames(df) <- paste0("Gene_", 1:nrow(df))
  }
}

# Print dataset information
cat("Original dataset dimensions:", nrow(df), "genes x", ncol(df), "samples\n")
cat("Gene names preview (internal only):", paste(head(rownames(df), 5), collapse = ", "), "...\n")

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

# Function to save heatmap in multiple formats - WITHOUT GENE NAMES, with Times New Roman font and TIGHT LAYOUT
save_heatmap_multiple_formats_no_genes <- function(data, base_filename, title, output_directory = output_dir, 
                                                   fontsize_col = 12, 
                                                   width = NULL, height = NULL) {
  
  # File paths for different formats
  png_file <- file.path(output_directory, paste0(base_filename, ".png"))
  jpg_file <- file.path(output_directory, paste0(base_filename, ".jpg"))
  pdf_file <- file.path(output_directory, paste0(base_filename, ".pdf"))
  
  cat("Saving heatmap in multiple formats WITHOUT gene names, with Times New Roman font and tight layout...\n")
  
  # Calculate optimal dimensions based on data size (more compact without gene names)
  if (is.null(width)) {
    width <- max(6, min(18, ncol(data) * 0.25 + 3))  # Smaller width since no gene names
  }
  if (is.null(height)) {
    height <- max(4, min(14, nrow(data) * 0.2 + 2))  # Smaller height since no gene names
  }
  
  cat("Using dimensions:", width, "x", height, "inches\n")
  
  # Set font family to Times New Roman
  par(family = "serif")  # serif = Times New Roman family
  
  # PNG format (high DPI) - WITHOUT GENE NAMES, with Times New Roman and TIGHT LAYOUT
  png(png_file, width = width, height = height, units = "in", res = HIGH_DPI, family = "serif")
  pheatmap(
    data,
    show_rownames = FALSE,     # HIDE gene names
    show_colnames = TRUE,      # Show sample names
    fontsize_col = fontsize_col,
    main = title,
    clustering_distance_rows = "euclidean",
    clustering_distance_cols = "euclidean",
    clustering_method = "complete",
    border_color = "white",    # Add borders for better separation
    cellwidth = 18,           # Slightly smaller cells since no gene names
    cellheight = 15,          # Compact cell height
    fontfamily = "serif",     # Times New Roman font family
    # TIGHT LAYOUT SETTINGS
    margin = c(1, 1),         # Minimal margins [bottom, right]
    treeheight_row = 40,      # Smaller row dendrogram
    treeheight_col = 40       # Smaller column dendrogram
  )
  dev.off()
  cat("✓ Saved PNG without gene names (tight layout):", png_file, "\n")
  
  # JPEG format (high DPI, high quality) - WITHOUT GENE NAMES
  jpeg(jpg_file, width = width, height = height, units = "in", res = HIGH_DPI, quality = 95, family = "serif")
  pheatmap(
    data,
    show_rownames = FALSE,     # HIDE gene names
    show_colnames = TRUE,      # Show sample names
    fontsize_col = fontsize_col,
    main = title,
    clustering_distance_rows = "euclidean",
    clustering_distance_cols = "euclidean",
    clustering_method = "complete",
    border_color = "white",    # Add borders for better separation
    cellwidth = 18,           # Slightly smaller cells since no gene names
    cellheight = 15,          # Compact cell height
    fontfamily = "serif",     # Times New Roman font family
    # TIGHT LAYOUT SETTINGS
    margin = c(1, 1),         # Minimal margins [bottom, right]
    treeheight_row = 40,      # Smaller row dendrogram
    treeheight_col = 40       # Smaller column dendrogram
  )
  dev.off()
  cat("✓ Saved JPEG without gene names (tight layout):", jpg_file, "\n")
  
  # PDF format (vector, scalable) - WITHOUT GENE NAMES
  pdf(pdf_file, width = width, height = height, family = "serif")
  pheatmap(
    data,
    show_rownames = FALSE,     # HIDE gene names
    show_colnames = TRUE,      # Show sample names
    fontsize_col = fontsize_col,
    main = title,
    clustering_distance_rows = "euclidean",
    clustering_distance_cols = "euclidean",
    clustering_method = "complete",
    border_color = "white",    # Add borders for better separation
    cellwidth = 18,           # Slightly smaller cells since no gene names
    cellheight = 15,          # Compact cell height
    fontfamily = "serif",     # Times New Roman font family
    # TIGHT LAYOUT SETTINGS
    margin = c(1, 1),         # Minimal margins [bottom, right]
    treeheight_row = 40,      # Smaller row dendrogram
    treeheight_col = 40       # Smaller column dendrogram
  )
  dev.off()
  cat("✓ Saved PDF without gene names (tight layout):", pdf_file, "\n")
  
  # Reset font family
  par(family = "")
  
  return(list(png = png_file, jpg = jpg_file, pdf = pdf_file))
}

# Create the main heatmap WITHOUT gene names
main_title <- paste("Gene Expression Heatmap -", GSE_NUMBER, "\n(", nrow(Data1), "genes x", ncol(Data1), "samples)")

# Adjust font sizes (only for column names since no row names)
fontsize_col <- ifelse(ncol(Data1) > 30, 8, ifelse(ncol(Data1) > 20, 10, 12))

cat("Using font size for sample names:", fontsize_col, "\n")

# Display heatmap in R console/viewer - WITHOUT GENE NAMES, with Times New Roman font and TIGHT LAYOUT
cat("Displaying heatmap WITHOUT gene names, with Times New Roman font and tight layout...\n")

# Set font family to Times New Roman for display
par(family = "serif")

pheatmap(
  Data1, 
  show_rownames = FALSE,     # HIDE gene names
  show_colnames = TRUE,      # Show sample names on top
  fontsize_col = fontsize_col,
  main = main_title,
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  clustering_method = "complete",
  border_color = "white",    # Add white borders for better separation
  cellwidth = 18,           # Compact cell width
  cellheight = 15,          # Compact cell height
  fontfamily = "serif",     # Times New Roman font family
  # TIGHT LAYOUT SETTINGS for display
  margin = c(1, 1),         # Minimal margins [bottom, right]
  treeheight_row = 40,      # Smaller row dendrogram
  treeheight_col = 40       # Smaller column dendrogram
)

# Save main heatmap in all formats WITHOUT gene names
main_files <- save_heatmap_multiple_formats_no_genes(
  Data1, 
  paste0(GSE_NUMBER, "_main_heatmap_no_gene_names"), 
  main_title,
  fontsize_col = fontsize_col
)

# ENHANCED FUNCTION: Create multiple heatmaps with different sample subsets - WITHOUT GENE NAMES
create_subset_heatmaps_no_genes <- function(data, samples_per_plot = 20, gse_id = GSE_NUMBER, 
                                            output_directory = output_dir) {
  
  # Create subdirectory for subset heatmaps
  subset_dir <- file.path(output_directory, "subset_heatmaps")
  if (!dir.exists(subset_dir)) {
    dir.create(subset_dir, recursive = TRUE)
  }
  
  total_samples <- ncol(data)
  num_plots <- ceiling(total_samples / samples_per_plot)
  
  cat("\nCreating", num_plots, "subset heatmaps WITHOUT gene names with", samples_per_plot, "samples each\n")
  
  # Adjust font sizes for subsets (only column names)
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
    base_filename <- paste0(gse_id, "_subset_", i, "_samples_", start_col, "_to_", end_col, "_no_gene_names")
    
    # Save in all formats with tight layout WITHOUT gene names
    subset_files <- save_heatmap_multiple_formats_no_genes(
      subset_data,
      base_filename,
      subset_title,
      output_directory = subset_dir,
      fontsize_col = subset_fontsize_col,
      width = NULL,  # Let function calculate optimal width
      height = NULL  # Let function calculate optimal height
    )
    
    cat("Subset", i, "completed (no gene names)\n")
  }
  
  cat("All subset heatmaps WITHOUT gene names saved in:", subset_dir, "\n")
}

# Create subset heatmaps WITHOUT gene names
create_subset_heatmaps_no_genes(as.matrix(df), samples_per_plot = 25)

# BONUS: Create a summary report
create_summary_report_no_genes <- function(original_data, limited_data, gse_id = GSE_NUMBER, 
                                           output_directory = output_dir) {
  
  report_file <- file.path(output_directory, paste0(gse_id, "_analysis_summary_no_gene_names.txt"))
  
  # Prepare summary information
  summary_text <- paste(
    "=== GENE EXPRESSION HEATMAP ANALYSIS SUMMARY (NO GENE NAMES) ===",
    paste("Dataset:", gse_id),
    paste("Analysis Date:", Sys.Date()),
    paste("Analysis Time:", Sys.time()),
    "",
    "=== DATA INFORMATION ===",
    paste("Original dimensions:", nrow(original_data), "genes x", ncol(original_data), "samples"),
    paste("Limited dimensions:", nrow(limited_data), "genes x", ncol(limited_data), "samples"),
    paste("Sample selection method: Random sampling with seed 123"),
    "",
    "=== VISUALIZATION SETTINGS ===",
    "Gene names: HIDDEN (not displayed on heatmaps)",
    "Sample names: VISIBLE (displayed on top)",
    paste("Total genes analyzed:", nrow(limited_data), "(gene names stored internally but not shown)"),
    "",
    "=== SAMPLE INFORMATION ===",
    paste("Total samples visualized:", ncol(limited_data)),
    paste("Sample names:", paste(head(colnames(limited_data), 10), collapse = ", ")),
    if(ncol(limited_data) > 10) "..." else "",
    "",
    "=== FILES GENERATED ===",
    "Main heatmap files (no gene names):",
    paste("- PNG:", paste0(gse_id, "_main_heatmap_no_gene_names.png")),
    paste("- JPEG:", paste0(gse_id, "_main_heatmap_no_gene_names.jpg")),
    paste("- PDF:", paste0(gse_id, "_main_heatmap_no_gene_names.pdf")),
    "",
    "Subset heatmaps (no gene names): Located in 'subset_heatmaps' subdirectory",
    "",
    "=== ANALYSIS PARAMETERS ===",
    paste("Maximum samples per plot:", MAX_SAMPLES),
    paste("Image resolution (DPI):", HIGH_DPI),
    paste("Clustering method: Complete linkage"),
    paste("Distance metric: Euclidean"),
    "Font family: Times New Roman (serif)",
    "Gene names display: DISABLED",
    "",
    sep = "\n"
  )
  
  # Write summary to file
  writeLines(summary_text, report_file)
  cat("Summary report (no gene names version) saved:", report_file, "\n")
}

# Generate summary report
create_summary_report_no_genes(df, Data1)

cat("\n=== ANALYSIS COMPLETE (NO GENE NAMES VERSION) ===\n")
cat("All files saved in directory:", output_dir, "\n")
cat("✓ Main heatmap WITHOUT gene names (PNG, JPEG, PDF)\n")
cat("✓ Subset heatmaps WITHOUT gene names in 'subset_heatmaps' folder\n")
cat("✓ Analysis summary report\n")
cat("✓ Gene names are HIDDEN on all heatmaps\n")
cat("✓ Heatmaps are more compact due to no gene name labels\n")

# Display final file structure
cat("\nGenerated file structure:\n")
if (file.exists(output_dir)) {
  files <- list.files(output_dir, recursive = TRUE, full.names = FALSE)
  cat(paste(files, collapse = "\n"))
} else {
  cat("Output directory not found")
}
# Calculate Euclidean Distance Matrix from Gene Expression Data

# Install required packages if not already installed
if (!require(readr)) install.packages("readr")
if (!require(dplyr)) install.packages("dplyr")
if (!require(pheatmap)) install.packages("pheatmap")
if (!require(RColorBrewer)) install.packages("RColorBrewer")

library(readr)
library(dplyr)
library(pheatmap)
library(RColorBrewer)

# CONFIGURATION PARAMETERS - EDIT THESE AS NEEDED
GSE_NUMBER <- "GSE33630"           # Change this to your GSE number
PREVIEW_SAMPLE_SIZE <- 20          # Number of samples to use in preview heatmap
CREATE_PREVIEW <- TRUE             # Set to FALSE to skip preview heatmap creation
HIGH_DPI <- 1000                    # DPI for high-quality images

# Function to calculate euclidean distance matrix
calculate_euclidean_distance_matrix <- function(gse_number, preview_size = 100, create_preview = TRUE, dpi = 300) {
  
  # Create output directory
  output_dir <- paste0(gse_number, "_analysis")
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    cat("Created directory:", output_dir, "\n")
  }
  
  # Generate file paths (all files go in the GSE folder)
  input_file <- paste0(gse_number, "_matrix_Transpose_labeled.csv")
  output_file <- file.path(output_dir, paste0(gse_number, "_euclidean_distance_matrix.csv"))
  
  # High-quality image files in multiple formats
  preview_png <- file.path(output_dir, paste0(gse_number, "_preview_heatmap.png"))
  preview_jpg <- file.path(output_dir, paste0(gse_number, "_preview_heatmap.jpg"))
  preview_pdf <- file.path(output_dir, paste0(gse_number, "_preview_heatmap.pdf"))
  
  cat("=== EUCLIDEAN DISTANCE MATRIX CALCULATION ===\n")
  cat("GSE Number:", gse_number, "\n")
  cat("Output directory:", output_dir, "\n")
  cat("Input file:", input_file, "\n")
  cat("Output file:", output_file, "\n")
  cat("Preview sample size:", preview_size, "\n")
  cat("Create preview heatmap:", create_preview, "\n")
  cat("Image DPI:", dpi, "\n\n")
  
  # Check if input file exists
  if (!file.exists(input_file)) {
    stop(paste("Input file not found:", input_file))
  }
  
  cat("Reading data from:", input_file, "\n")
  
  # Read the CSV file
  data <- read_csv(input_file, show_col_types = FALSE)
  
  cat("Data dimensions:", nrow(data), "samples,", ncol(data), "columns\n")
  
  # Assuming the last column is the target variable
  # Remove the target column to get only gene expression data
  gene_data <- data[, -ncol(data)]  # Remove last column (target)
  target_data <- data[, ncol(data)] # Keep target for reference
  
  cat("Gene expression data dimensions:", nrow(gene_data), "samples,", ncol(gene_data), "genes\n")
  
  # Convert to matrix (samples as rows, genes as columns)
  gene_matrix <- as.matrix(gene_data)
  
  # Handle missing values by replacing with column means
  missing_count <- sum(is.na(gene_matrix))
  if (missing_count > 0) {
    cat("Found", missing_count, "missing values. Replacing with column means...\n")
    for(i in 1:ncol(gene_matrix)) {
      col_mean <- mean(gene_matrix[, i], na.rm = TRUE)
      gene_matrix[is.na(gene_matrix[, i]), i] <- col_mean
    }
  }
  
  cat("Calculating euclidean distance matrix...\n")
  
  # Calculate euclidean distance matrix between samples (rows)
  # dist() function calculates distances between rows by default
  distance_matrix <- dist(gene_matrix, method = "euclidean")
  
  # Convert to full matrix format
  distance_matrix_full <- as.matrix(distance_matrix)
  
  # Add row and column names
  sample_names <- paste0("Sample_", 1:nrow(gene_matrix))
  rownames(distance_matrix_full) <- sample_names
  colnames(distance_matrix_full) <- sample_names
  
  cat("Distance matrix dimensions:", nrow(distance_matrix_full), "x", ncol(distance_matrix_full), "\n")
  
  # Display summary statistics
  cat("\nSummary Statistics:\n")
  distances <- distance_matrix_full[upper.tri(distance_matrix_full)]
  cat("Min distance:", round(min(distances), 4), "\n")
  cat("Max distance:", round(max(distances), 4), "\n")
  cat("Mean distance:", round(mean(distances), 4), "\n")
  cat("Median distance:", round(median(distances), 4), "\n")
  
  # Display target distribution
  target_table <- table(data[[ncol(data)]])
  cat("\nTarget distribution:\n")
  print(target_table)
  
  # Save the distance matrix
  cat("\nSaving distance matrix to:", output_file, "\n")
  
  # Convert matrix to data frame with row names as first column
  distance_df <- data.frame(Sample = rownames(distance_matrix_full), 
                            distance_matrix_full, 
                            check.names = FALSE)
  
  # Write to CSV
  write_csv(distance_df, output_file)
  
  cat("Distance matrix saved successfully!\n")
  
  # Create preview heatmap if requested
  if (create_preview) {
    cat("\nCreating preview heatmaps in multiple formats...\n")
    
    # Determine sample size for preview
    total_samples <- nrow(distance_matrix_full)
    actual_preview_size <- min(preview_size, total_samples)
    
    if (total_samples > preview_size) {
      cat("Sampling", actual_preview_size, "out of", total_samples, "samples for preview\n")
      # Sample indices for preview
      set.seed(42)  # For reproducible sampling
      sample_indices <- sample(1:total_samples, actual_preview_size)
      preview_matrix <- distance_matrix_full[sample_indices, sample_indices]
    } else {
      cat("Using all", total_samples, "samples for preview\n")
      preview_matrix <- distance_matrix_full
    }
    
    # Define color palette
    colors <- colorRampPalette(brewer.pal(9, "YlOrBr"))(100)
    
    # Create heatmap title
    heatmap_title <- paste("Euclidean Distance Heatmap -", gse_number, 
                           "\n(", actual_preview_size, "samples)")
    
    # Save as PNG (high DPI)
    cat("Saving PNG format...\n")
    png(preview_png, width = 10, height = 8, units = "in", res = dpi)
    pheatmap(preview_matrix,
             clustering_distance_rows = "euclidean",
             clustering_distance_cols = "euclidean", 
             clustering_method = "complete",
             main = heatmap_title,
             show_colnames = FALSE,
             show_rownames = FALSE,
             color = colors)
    dev.off()
    
    # Save as JPEG (high DPI)
    cat("Saving JPEG format...\n")
    jpeg(preview_jpg, width = 10, height = 8, units = "in", res = dpi, quality = 95)
    pheatmap(preview_matrix,
             clustering_distance_rows = "euclidean",
             clustering_distance_cols = "euclidean", 
             clustering_method = "complete",
             main = heatmap_title,
             show_colnames = FALSE,
             show_rownames = FALSE,
             color = colors)
    dev.off()
    
    # Save as PDF (vector format)
    cat("Saving PDF format...\n")
    pdf(preview_pdf, width = 10, height = 8)
    pheatmap(preview_matrix,
             clustering_distance_rows = "euclidean",
             clustering_distance_cols = "euclidean", 
             clustering_method = "complete",
             main = heatmap_title,
             show_colnames = FALSE,
             show_rownames = FALSE,
             color = colors)
    dev.off()
    
    cat("Preview heatmaps saved in 3 formats:\n")
    cat("- PNG (high DPI):", preview_png, "\n")
    cat("- JPEG (high DPI):", preview_jpg, "\n")
    cat("- PDF (vector):", preview_pdf, "\n")
  }
  
  # Return the matrix for further analysis if needed
  return(list(
    distance_matrix = distance_matrix_full,
    target_data = target_data,
    sample_names = sample_names,
    input_file = input_file,
    output_file = output_file,
    output_dir = output_dir,
    preview_files = if(create_preview) list(png = preview_png, jpg = preview_jpg, pdf = preview_pdf) else NULL
  ))
}

# MAIN EXECUTION
cat("Starting distance matrix calculation...\n\n")

# Calculate and save the distance matrix
tryCatch({
  result <- calculate_euclidean_distance_matrix(
    gse_number = GSE_NUMBER,
    preview_size = PREVIEW_SAMPLE_SIZE,
    create_preview = CREATE_PREVIEW,
    dpi = HIGH_DPI
  )
  
  cat("\n=== SCRIPT COMPLETED SUCCESSFULLY ===\n")
  cat("Output directory:", result$output_dir, "\n")
  cat("Files generated:\n")
  cat("- Distance matrix CSV:", basename(result$output_file), "\n")
  if (CREATE_PREVIEW) {
    cat("- Preview heatmap PNG:", basename(result$preview_files$png), "\n")
    cat("- Preview heatmap JPEG:", basename(result$preview_files$jpg), "\n")
    cat("- Preview heatmap PDF:", basename(result$preview_files$pdf), "\n")
  }
  cat("\nAll files are saved in the", result$output_dir, "folder\n")
  cat("You can now use the distance matrix file with your heatmap visualization code!\n")
  
  # Optional: Display first few rows/columns of the distance matrix
  cat("\nFirst 5x5 section of distance matrix:\n")
  print(round(result$distance_matrix[1:5, 1:5], 3))
  
}, error = function(e) {
  cat("Error occurred:", e$message, "\n")
  cat("Please check that your input file exists and is properly formatted.\n")
  cat("Expected input file:", paste0(GSE_NUMBER, "_matrix_Transpose_labeled.csv"), "\n")
})

# ADDITIONAL FUNCTION: Create full heatmap with all samples
create_full_heatmap <- function(gse_number, dpi = 1000) {
  output_dir <- paste0(gse_number, "_analysis")
  input_file <- paste0(gse_number, "_euclidean_distance_matrix.csv")
  
  if (!file.exists(file.path(output_dir, basename(input_file)))) {
    cat("Distance matrix file not found. Please run the main script first.\n")
    return(NULL)
  }
  
  # Read the distance matrix
  distance_df <- read_csv(file.path(output_dir, basename(input_file)), show_col_types = FALSE)
  distance_matrix <- as.matrix(distance_df[, -1])
  rownames(distance_matrix) <- distance_df$Sample
  
  # File paths for full heatmap
  full_png <- file.path(output_dir, paste0(gse_number, "_full_heatmap.png"))
  full_jpg <- file.path(output_dir, paste0(gse_number, "_full_heatmap.jpg"))
  full_pdf <- file.path(output_dir, paste0(gse_number, "_full_heatmap.pdf"))
  
  colors <- colorRampPalette(brewer.pal(9, "YlOrBr"))(100)
  heatmap_title <- paste("Full Euclidean Distance Heatmap -", gse_number, 
                         "\n(All", nrow(distance_matrix), "samples)")
  
  cat("Creating full heatmap with all samples...\n")
  
  # PNG
  png(full_png, width = 12, height = 10, units = "in", res = dpi)
  pheatmap(distance_matrix,
           clustering_distance_rows = "euclidean",
           clustering_distance_cols = "euclidean",
           clustering_method = "complete",
           main = heatmap_title,
           show_colnames = FALSE,
           show_rownames = FALSE,
           color = colors)
  dev.off()
  
  # JPEG
  jpeg(full_jpg, width = 12, height = 10, units = "in", res = dpi, quality = 95)
  pheatmap(distance_matrix,
           clustering_distance_rows = "euclidean",
           clustering_distance_cols = "euclidean",
           clustering_method = "complete",
           main = heatmap_title,
           show_colnames = FALSE,
           show_rownames = FALSE,
           color = colors)
  dev.off()
  
  # PDF
  pdf(full_pdf, width = 12, height = 10)
  pheatmap(distance_matrix,
           clustering_distance_rows = "euclidean",
           clustering_distance_cols = "euclidean",
           clustering_method = "complete",
           main = heatmap_title,
           show_colnames = FALSE,
           show_rownames = FALSE,
           color = colors)
  dev.off()
  
  cat("Full heatmaps saved:\n")
  cat("- PNG:", full_png, "\n")
  cat("- JPEG:", full_jpg, "\n")
  cat("- PDF:", full_pdf, "\n")
}

create_full_heatmap(GSE_NUMBER, HIGH_DPI)

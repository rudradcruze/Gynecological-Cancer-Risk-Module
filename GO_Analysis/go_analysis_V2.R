# Install necessary packages if not already installed
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("clusterProfiler", force = TRUE)
BiocManager::install("org.Hs.eg.db")
BiocManager::install("enrichplot")
BiocManager::install("ggplot2")

# Load the libraries
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(ggplot2)

# Update packages
update.packages(ask = FALSE, checkBuilt = TRUE)

# Define the gene list
genes <- c("TK1", "RHEB", "ETFBKMT", "SEC16B", "MIR664B///SNORA56///DKC1", "MSH6", "TTK", "E2F1", "MDM2", "SLC3A1",  
           "KIF14", "CENPF", "MCM4", "RARG", "ASF1B", "KIF4A", "USP21", "ATAD2", "CENPU", "GCC1",  
           "ASPM", "CCDC30", "TNS1", "MCM4.2", "GMEB2", "MGAT4B", "TVP23C-CDRT4///CDRT4///TVP23C", "UBXN10", "BRIP1", "TNRC6C")

# Define the ontology type (change this to "BP", "CC", or "ALL" as needed)
ontology_type <- "CC"  # Current setting: Molecular Function

# Create folder based on ontology type
folder_name <- ontology_type
if (!dir.exists(folder_name)) {
  dir.create(folder_name, recursive = TRUE)
  cat("Created folder:", folder_name, "\n")
} else {
  cat("Folder already exists:", folder_name, "\n")
}

# Perform GO enrichment analysis
go_enrichment <- enrichGO(gene         = genes,
                          OrgDb        = org.Hs.eg.db,
                          keyType      = "SYMBOL",  # Indicating the key type is SYMBOL
                          ont          = ontology_type,    # Using the defined ontology type
                          pAdjustMethod = "BH",    # Benjamini-Hochberg correction for multiple testing
                          pvalueCutoff = 0.25,
                          qvalueCutoff = 0.25
)

# View and save GO enrichment results to CSV in the appropriate folder
go_enrichment_df <- as.data.frame(go_enrichment)
csv_file <- file.path(folder_name, paste0("GO_Enrichment_Results_", ontology_type, ".csv"))
write.csv(go_enrichment_df, file = csv_file, row.names = FALSE)
cat("CSV saved to:", csv_file, "\n")

# Define the title based on ontology type
ontology_titles <- list(
  "BP" = "Biological Process (BP)",
  "MF" = "Molecular Function (MF)",
  "CC" = "Cellular Component (CC)",
  "ALL" = "All GO Categories"
)
title <- ontology_titles[[ontology_type]]

# Save Dotplot
dotplot_plot <- dotplot(go_enrichment, showCategory = 15) +
  ggtitle(title) +
  theme(plot.title = element_text(hjust = 0.5))
dotplot_plot

# Save dotplot files in the folder
ggsave(file.path(folder_name, paste0("Dotplot_GO_", ontology_type, ".jpeg")), 
       plot = dotplot_plot, width = 7, height = 8, dpi = 1000)
ggsave(file.path(folder_name, paste0("Dotplot_GO_", ontology_type, ".png")), 
       plot = dotplot_plot, width = 7, height = 8, dpi = 1000)
ggsave(file.path(folder_name, paste0("Dotplot_GO_", ontology_type, ".pdf")), 
       plot = dotplot_plot, width = 7, height = 8)

# Save Barplot
barplot_plot <- barplot(go_enrichment, showCategory = 15) +
  ggtitle(title) +
  theme(plot.title = element_text(hjust = 0.5))
barplot_plot

# Save barplot files in the folder
ggsave(file.path(folder_name, paste0("Barplot_GO_", ontology_type, ".jpeg")), 
       plot = barplot_plot, width = 7, height = 8, dpi = 1000)
ggsave(file.path(folder_name, paste0("Barplot_GO_", ontology_type, ".png")), 
       plot = barplot_plot, width = 7, height = 8, dpi = 1000)
ggsave(file.path(folder_name, paste0("Barplot_GO_", ontology_type, ".pdf")), 
       plot = barplot_plot, width = 7, height = 8)

# Save Cnetplot
cnetplot_plot <- cnetplot(go_enrichment, showCategory = 10) +
  ggtitle("GO Term Network") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.margin = margin(20, 20, 20, 20)
  )
cnetplot_plot

# Save cnetplot files in the folder
ggsave(file.path(folder_name, paste0("Cnetplot_GO_", ontology_type, ".jpeg")), 
       plot = cnetplot_plot, width = 12, height = 10, dpi = 1000)
ggsave(file.path(folder_name, paste0("Cnetplot_GO_", ontology_type, ".png")), 
       plot = cnetplot_plot, width = 12, height = 10, dpi = 1000)
ggsave(file.path(folder_name, paste0("Cnetplot_GO_", ontology_type, ".pdf")), 
       plot = cnetplot_plot, width = 12, height = 10)

# Calculate term similarity and Save Emapplot
go_enrichment_sim <- pairwise_termsim(go_enrichment)
emapplot_plot <- emapplot(go_enrichment_sim, showCategory = 10) +
  ggtitle("GO Term Enrichment Map") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.margin = margin(20, 20, 20, 20)
  )
emapplot_plot

# Save emapplot files in the folder
ggsave(file.path(folder_name, paste0("Emapplot_GO_", ontology_type, ".jpeg")), 
       plot = emapplot_plot, width = 12, height = 10, dpi = 1000)
ggsave(file.path(folder_name, paste0("Emapplot_GO_", ontology_type, ".png")), 
       plot = emapplot_plot, width = 12, height = 10, dpi = 1000)
ggsave(file.path(folder_name, paste0("Emapplot_GO_", ontology_type, ".pdf")), 
       plot = emapplot_plot, width = 12, height = 10)

cat("All plots and results saved in folder:", folder_name, "\n")


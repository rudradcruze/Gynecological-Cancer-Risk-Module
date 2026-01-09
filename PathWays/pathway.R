# Install required packages if not installed
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
if (!requireNamespace("enrichplot", quietly = TRUE)) {
  install.packages("enrichplot")
}
if (!requireNamespace("stringr", quietly = TRUE)) {
  install.packages("stringr")
}

# Load the libraries
library(ggplot2)
library(enrichplot)
library(stringr)  # Load stringr for text manipulation
library(scales)  # Load scales for formatting numbers

update.packages(ask = FALSE, checkBuilt = TRUE)

# Step 1: Read the CSV file
data <- read.csv("KEGG_2021_Human_table.csv")  # Replace with your actual CSV file path
graphics.off()

# Step 2: Calculate GeneRatio from the "Overlap" column
data$GeneRatio <- apply(data, 1, function(row) {
  overlap_split <- strsplit(as.character(row["Overlap"]), "/")[[1]]
  as.numeric(overlap_split[1]) / as.numeric(overlap_split[2])
})

# Step 3: Sort the data by Adjusted.P.value in ascending order
sorted_data <- data[order(data$Adjusted.P.value), ]

# Step 4: Select the top 5 rows after sorting
top_20_data <- head(sorted_data, 10)

# Step 5: Wrap y-axis labels for better visibility
top_20_data$Term <- str_wrap(top_20_data$Term, width = 37)  # Wrap the text at 30 characters

# Step 6: Create the dot plot and store it in a variable
kegg_plot <- ggplot(top_20_data, aes(x = "", y = reorder(Term, -GeneRatio), 
                                     size = GeneRatio, color = Adjusted.P.value)) +
  geom_point(shape = 16) +  # Use shape 16 for filled circles
  scale_color_gradient(low = "red", high = "blue", 
                       labels = number_format(accuracy = 0.01)) +  # Limit to 2 decimal places
  labs(title = "KEGG Pathway",
       x = "Cluster",  # Replace this label if needed
       y = NULL,  # No label on y-axis
       color = "p.adjust") +  # Legend for Adjusted P-value
  theme_minimal() +
  theme(axis.text.y = element_text(size = 10, color = "black"),
        plot.margin = margin(20, 20, 20, 20),
        plot.title = element_text(hjust = 0.5)) +
  theme(panel.border = element_rect(color = "black", fill = NA, size = 1.2))

# Display the plot
print(kegg_plot)

# Step 7: Save the plot in multiple formats
# Save in PDF format
ggsave("kegg_pathway_plot.pdf", 
       plot = kegg_plot,
       width = 10, 
       height = 10, 
       units = "in",
       device = "pdf")

# Save in PNG format
ggsave("kegg_pathway_plot.png", 
       plot = kegg_plot,
       width = 10, 
       height = 10, 
       units = "in",
       dpi = 1000)

# Save in JPG format
ggsave("kegg_pathway_plot.jpg", 
       plot = kegg_plot,
       width = 10, 
       height = 10, 
       units = "in",
       dpi = 1000)


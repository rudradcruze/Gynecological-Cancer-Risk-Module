# Gynecological Cancer Risk Module

A comprehensive bioinformatics pipeline for identifying gene expression patterns and risk factors associated with gynecological cancers using machine learning and feature selection techniques.

## Table of Contents

-   [Overview](#overview)
-   [Project Structure](#project-structure)
-   [Methodology](#methodology)
-   [Features](#features)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Modules](#modules)
-   [Authors](#authors)
-   [Citation](#citation)
-   [License](#license)

## Overview

This project implements a machine learning pipeline for gynecological cancer risk assessment using gene expression data from publicly available datasets. The pipeline incorporates advanced feature selection methods and deep learning models (Attention-based Bidirectional LSTM) to identify and validate predictive biomarkers for cancer risk classification.

### Key Objectives

-   Analyze gene expression profiles from gynecological cancer datasets
-   Perform comprehensive feature selection using multiple methods (SHAP, Lasso, ANOVA, MIG, RFE, Boruta, Lime)
-   Develop and evaluate attention-based deep learning models for cancer risk classification
-   Conduct pathway and gene ontology analysis to understand biological mechanisms
-   Generate publication-ready visualizations and statistical analyses

## Project Structure

```
Gynecological-Cancer-Risk-Module/
├── 0_Dataset_Info/                      # Dataset information and exploration
│   └── 0_Dataset_Info.ipynb
├── 1_Feature_Subset_Without_FS/         # Baseline models without feature selection
│   ├── 1_Feature_Subset_Without_FS.ipynb
│   ├── 2_Feature_Subset_With_FS.ipynb
│   ├── 2_Feature_Subset_With_FS_Without_Early_Stop.ipynb
│   └── 3_Feature_Subset_With_FS_2.ipynb
├── 2_Feature_Subset_With_FS/            # Feature selection implementations
│   ├── Boruta/
│   │   ├── stage1_boruta_calculation.py
│   │   ├── stage2_feature_selection_plot_column.py
│   │   └── stage3_feature_selection.py
│   └── Lime/
│       ├── stage1_lime_calculation.py
│       ├── stage2_feature_selection_plot_column.py
│       └── stage3_feature_selection.py
├── 3_Model_Comparison/                  # Cross-model evaluation
│   └── 1_Model_Comparison.ipynb
├── 4_Feature_Selection/                 # Advanced feature selection methods
│   ├── 1_SHAP.py                       # SHAP-based feature importance
│   ├── 2_Laso.py                       # Lasso feature selection
│   ├── 3_Anova.py                      # ANOVA-based feature selection
│   ├── 4_mig.py                        # Mutual Information Gain
│   └── 5_RFE.py                        # Recursive Feature Elimination
├── Euclidean_Heatmap/                   # Euclidean distance-based analysis
│   ├── Distance_Calculate-and-Heatmap.R
│   └── heatmap.R
├── GO_Analysis/                         # Gene Ontology enrichment analysis
│   ├── go_analysis.R
│   └── go_analysis_V2.R
├── HeatMap/                             # Expression heatmap visualizations
│   ├── Heatmap.R
│   ├── Heatmap-With-GENE-Names.R
│   └── Heatmap-Without-GENE-Names.R
├── PathWays/                            # KEGG pathway analysis
│   └── pathway.R
├── RMA_Box_Plot/                        # Expression distribution analysis
│   └── box_plot.R
└── README.md
```

## Methodology

### 1. **Data Preparation & Normalization**

-   Load expression datasets from GEO (Gene Expression Omnibus)
-   Data quality control and normalization
-   Target variable encoding and stratification

### 2. **Feature Selection (Multiple Approaches)**

**Traditional Statistical Methods:**

-   ANOVA (Analysis of Variance)
-   Mutual Information Gain (MIG)

**Machine Learning Methods:**

-   SHAP (SHapley Additive exPlanations) - interpretable feature importance
-   Lasso Regression - L1 regularization
-   RFE (Recursive Feature Elimination) - iterative feature removal
-   Boruta - feature selection via random forests
-   LIME - Local Interpretable Model-agnostic Explanations

### 3. **Deep Learning Model**

-   **Architecture**: Attention-based Bidirectional LSTM (AttBiLSTM)
-   **Input**: Gene expression features
-   **Output**: Binary/Multi-class cancer risk classification
-   **Training**: Stratified k-fold cross-validation, early stopping
-   **Optimization**: Adam optimizer with mixed precision training

### 4. **Biological Analysis**

-   **Gene Ontology (GO)**: Functional annotation and enrichment
-   **KEGG Pathways**: Metabolic and signaling pathway analysis
-   **Heatmaps**: Expression pattern visualization with hierarchical clustering
-   **Distance Analysis**: Euclidean distance-based sample relationships

### 5. **Model Evaluation**

-   **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Matthews Correlation Coefficient
-   **Cross-validation**: Stratified K-Fold validation
-   **Visualization**: ROC curves, confusion matrices, feature importance plots

## Features

✅ Multi-method feature selection with comparative analysis  
✅ Attention-based deep learning for improved interpretability  
✅ GPU/MPS acceleration support (CUDA, Apple Metal)  
✅ Comprehensive cross-validation and evaluation metrics  
✅ Publication-ready visualizations  
✅ Biological pathway and GO term enrichment  
✅ Interactive Jupyter notebooks for reproducibility  
✅ Modular design for easy extension and adaptation

## Installation

### Requirements

-   Python 3.8+
-   R 4.0+
-   CUDA 11.0+ (optional, for GPU acceleration)

### Python Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install torch torchvision torchaudio
pip install shap boruta-py lime joblib tqdm
pip install scipy statsmodels
```

### R Dependencies

```R
install.packages(c("GEOquery", "limma", "heatmap3", "igraph"))
BiocManager::install(c("clusterProfiler", "enrichplot", "org.Hs.eg.db"))
install.packages("ggplot2")
```

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd Gynecological-Cancer-Risk-Module
```

2. Create a virtual environment (Python):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create data directory:

```bash
mkdir dataset
# Add your CSV files to this directory
```

## Usage

### 1. **Dataset Exploration** (Optional)

```bash
jupyter notebook 0_Dataset_Info/0_Dataset_Info.ipynb
```

### 2. **Baseline Model (No Feature Selection)**

```bash
jupyter notebook 1_Feature_Subset_Without_FS/1_Feature_Subset_Without_FS.ipynb
```

### 3. **Feature Selection Pipelines**

**SHAP-based Feature Selection:**

```bash
python 4_Feature_Selection/1_SHAP.py
```

Output: `AttBiLSTM_Analysis_With_SHAP_FS/`

**Lasso Feature Selection:**

```bash
python 4_Feature_Selection/2_Laso.py
```

Output: `AttBiLSTM_Analysis_With_Lasso_FS/`

**ANOVA Feature Selection:**

```bash
python 4_Feature_Selection/3_Anova.py
```

Output: `AttBiLSTM_Analysis_With_Anova_FS/`

**Mutual Information Gain:**

```bash
python 4_Feature_Selection/4_mig.py
```

Output: `AttBiLSTM_Analysis_With_MIG_FS/`

**RFE Feature Selection:**

```bash
python 4_Feature_Selection/5_RFE.py
```

Output: `AttBiLSTM_Analysis_With_RFE_FS/`

**Boruta Feature Selection:**

```bash
python 2_Feature_Subset_With_FS/Boruta/stage1_boruta_calculation.py
python 2_Feature_Subset_With_FS/Boruta/stage2_feature_selection_plot_column.py
python 2_Feature_Subset_With_FS/Boruta/stage3_feature_selection.py
```

**LIME Feature Selection:**

```bash
python 2_Feature_Subset_With_FS/Lime/stage1_lime_calculation.py
python 2_Feature_Subset_With_FS/Lime/stage2_feature_selection_plot_column.py
python 2_Feature_Subset_With_FS/Lime/stage3_feature_selection.py
```

### 4. **Model Comparison**

```bash
jupyter notebook 3_Model_Comparison/1_Model_Comparison.ipynb
```

### 5. **Biological Analysis** (R scripts)

**Heatmap Analysis:**

```bash
Rscript HeatMap/Heatmap.R
Rscript HeatMap/Heatmap-With-GENE-Names.R
Rscript HeatMap/Heatmap-Without-GENE-Names.R
```

**Gene Ontology Analysis:**

```bash
Rscript GO_Analysis/go_analysis.R
```

**Pathway Analysis:**

```bash
Rscript PathWays/pathway.R
```

**Distance-based Analysis:**

```bash
Rscript Euclidean_Heatmap/Distance_Calculate-and-Heatmap.R
```

## Modules

### 0_Dataset_Info

Explores dataset characteristics, dimensions, target distribution, and prepares data for modeling.

### 1_Feature_Subset_Without_FS

Implements baseline AttBiLSTM models without feature selection for comparison purposes.

### 2_Feature_Subset_With_FS

Contains Boruta and LIME feature selection implementations with three-stage pipelines:

-   **Stage 1**: Calculate feature importance scores
-   **Stage 2**: Generate visualizations and plots
-   **Stage 3**: Apply final feature selection

### 3_Model_Comparison

Compares performance across different models and feature selection strategies.

### 4_Feature_Selection

Advanced feature selection methods integrated with AttBiLSTM model training:

-   SHAP: Shapley value-based feature importance
-   Lasso: L1-regularized linear regression
-   ANOVA: Statistical hypothesis testing
-   MIG: Information-theoretic approach
-   RFE: Iterative model-based elimination

### Biological Analysis Modules

-   **GO_Analysis**: Gene Ontology enrichment using clusterProfiler
-   **HeatMap**: Expression pattern visualization
-   **PathWays**: KEGG pathway enrichment analysis
-   **Euclidean_Heatmap**: Sample distance clustering and visualization
-   **RMA_Box_Plot**: Distribution analysis of expression values

## Output Structure

Each feature selection method generates:

```
AttBiLSTM_Analysis_With_[METHOD]_FS/
├── models/                          # Saved model checkpoints (.pth)
├── plots/                           # Visualizations (ROC, confusion matrix, etc.)
├── results/                         # Metrics and evaluation results
└── feature_selection/               # Feature importance and selection results
```

## Authors

**Project Team:**

-   **Md. Mostary Khatun**  
    Student, Department of Information and Communication Technology (ICT)  
    Mawlana Bhashani Science and Technology University (MBSTU), Tangail, Bangladesh  
    Email: mostary.khatun09@gmail.com

-   **Francis Ridra D Cruze**  
    MS Student, Department of Computing and Information System  
    East West University, Aftabnagar Dhaka-1212, Bangladesh  
    Email: 2023-3-96-005@ewubd.edu

-   **Mst. Amina Khatun**  
    B.Sc. Student, Department of Computing and Information System  
    Daffodil International University, Savar, Dhaka-1216, Bangladesh  
    Email: amina32189101@diu.edu.bd

-   **Md. Farah Hosan**  
    Lecturer (Senior Scale), Department of Computing and Information System  
    Daffodil International University, Savar, Dhaka-1216, Bangladesh  
    Email: farah.cis@diu.edu.bd

-   **Muhammad Shahin Uddin**  
    Professor, Department of Information and Communication Technology (ICT)  
    Mawlana Bhashani Science and Technology University (MBSTU), Tangail, Bangladesh  
    Email: shahin.mbstu@gmail.com

-   **Monir Morshed**  
    Professor, Department of Information and Communication Technology (ICT)  
    Mawlana Bhashani Science and Technology University (MBSTU), Tangail, Bangladesh  
    Email: monir.morshed.ict@mbstu.ac.bd

-   **Md. Nasiml Kader**  
    Assistant Professor, Department of Computing and Information System  
    Daffodil International University, Savar, Dhaka-1216, Bangladesh  
    Email: nasiml.cis@diu.edu.bd

## Citation

Available Soon

## Requirements File (requirements.txt)

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.10.0
torchvision>=0.11.0
torchaudio>=0.10.0
shap>=0.40.0
boruta-py>=0.3.1
lime>=0.2.0
joblib>=1.1.0
tqdm>=4.62.0
scipy>=1.7.0
statsmodels>=0.13.0
```

## Data Sources

-   Gene Expression Omnibus (GEO): https://www.ncbi.nlm.nih.gov/geo/
-   Common datasets used:
    -   GSE33630: Ovarian cancer
    -   GSE5281: Breast/Ovarian cancer
    -   GSE120490: Gynecological cancer samples

## Key References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NIPS.
2. Kursa, M. B., & Rudnicki, W. R. (2010). Feature selection with the Boruta package. Journal of Statistical Software.
3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. KDD.
4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.
5. Vaswani, A., et al. (2017). Attention is all you need. NIPS.

## Troubleshooting

### CUDA/GPU Issues

If CUDA is not detected, the code will automatically fall back to CPU. To force CPU mode:

```python
device = torch.device("cpu")
```

### Memory Issues

For large datasets, consider:

-   Reducing batch size in model parameters
-   Implementing data generators instead of loading all data at once
-   Using feature selection to reduce dimensionality

### Missing Dependencies

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   Mawlana Bhashani Science and Technology University (MBSTU)
-   Daffodil International University
-   East West University
-   Gene Expression Omnibus (GEO) for providing public datasets
-   PyTorch and scikit-learn communities

---

**Last Updated**: January 2025  
**Version**: 1.0.0

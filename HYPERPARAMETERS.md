# Attention-based BiLSTM Model Hyperparameters

## Model Architecture Parameters

### Input Layer

-   **Input Shape**: (sequence_length, num_features)
-   **Sequence Length**: 100
-   **Number of Features**: Variable (depends on selected features from feature selection)

### BiLSTM Layer Configuration

-   **Units (Hidden Dimension)**: 128
-   **Return Sequences**: True
-   **Return State**: True
-   **Dropout**: 0.2
-   **Recurrent Dropout**: 0.2
-   **Activation Function**: 'tanh'
-   **Recurrent Activation**: 'sigmoid'

### Attention Layer

-   **Attention Type**: Multiplicative (Scaled Dot-Product)
-   **Attention Units**: 64
-   **Use Bias**: True

### Dense Layers (After Attention)

-   **Layer 1 Units**: 64
-   **Layer 1 Activation**: 'relu'
-   **Layer 1 Dropout**: 0.3

-   **Layer 2 Units**: 32
-   **Layer 2 Activation**: 'relu'
-   **Layer 2 Dropout**: 0.2

### Output Layer

-   **Units**: 1 (Binary Classification)
-   **Activation**: 'sigmoid'

## Training Hyperparameters

### Optimizer

-   **Type**: Adam
-   **Learning Rate**: 0.001
-   **Beta_1**: 0.9
-   **Beta_2**: 0.999
-   **Epsilon**: 1e-7
-   **Decay**: 0.0

### Loss Function

-   **Function**: Binary Crossentropy
-   **From Logits**: False

### Metrics

-   **Primary**: Binary Accuracy
-   **Secondary**: AUC, Precision, Recall, F1-Score

### Training Configuration

-   **Batch Size**: 32
-   **Epochs**: 100
-   **Validation Split**: 0.2 (20% of training data)
-   **Early Stopping Patience**: 15 epochs
-   **Early Stopping Monitor**: 'val_loss'
-   **Early Stopping Restore Best Weights**: True

### Class Weights (for imbalanced data)

-   **Calculated**: Yes (inversely proportional to class frequency)
-   **Applied During**: Training phase

## Data Preprocessing Hyperparameters

### Feature Scaling

-   **Method**: StandardScaler (Zero-mean, Unit-variance)
-   **Applied to**: Training data before model input

### Sequence Padding/Truncation

-   **Method**: Zero-padding (pre-padding)
-   **Target Length**: 100

### Train-Test Split

-   **Train Ratio**: 0.8 (80%)
-   **Test Ratio**: 0.2 (20%)
-   **Random State**: 42 (for reproducibility)

## Feature Selection Hyperparameters

### SelectKBest

-   **Score Function**: f_classif (ANOVA F-value)
-   **K (Number of Features)**: Variable (tested: 10, 15, 20, 25, 30)
-   **Default K**: 20

## Regularization Hyperparameters

### L1/L2 Regularization

-   **L2 Regularization (Kernel)**: 0.001
-   **L2 Regularization (Recurrent)**: 0.001

### Dropout Configuration

-   **BiLSTM Dropout**: 0.2
-   **Dense Layer 1 Dropout**: 0.3
-   **Dense Layer 2 Dropout**: 0.2

### Batch Normalization

-   **Applied**: After BiLSTM layer (optional configuration)
-   **Momentum**: 0.99
-   **Epsilon**: 0.001

## Evaluation Hyperparameters

### Threshold

-   **Classification Threshold**: 0.5
-   **Adjustable**: Yes (for ROC curve analysis)

### Cross-Validation

-   **Method**: K-Fold Cross-Validation
-   **K**: 5
-   **Shuffle**: True
-   **Random State**: 42

### Test Set Evaluation

-   **Metrics Calculated**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Specificity, Sensitivity

## Callback Configuration

### Early Stopping

-   **Monitor**: 'val_loss'
-   **Patience**: 15 epochs
-   **Min Delta**: 0.0001
-   **Mode**: 'min'
-   **Restore Best Weights**: True

### Model Checkpoint

-   **Save Best Only**: True
-   **Monitor**: 'val_auc'
-   **Mode**: 'max'

### Learning Rate Scheduler (Optional)

-   **Type**: ReduceLROnPlateau
-   **Monitor**: 'val_loss'
-   **Factor**: 0.5
-   **Patience**: 5 epochs
-   **Min LR**: 1e-7

## Hardware Configuration

### GPU/Computation

-   **GPU Usage**: TensorFlow auto-detect
-   **Mixed Precision Training**: False (Optional for faster training)

## Hyperparameter Tuning Search Space

### Tested Ranges

-   **BiLSTM Units**: [64, 128, 256]
-   **Learning Rate**: [0.0001, 0.001, 0.01]
-   **Batch Size**: [16, 32, 64]
-   **Dropout Rate**: [0.1, 0.2, 0.3, 0.4]
-   **Number of Features (K)**: [10, 15, 20, 25, 30]
-   **Dense Layer Units**: [32, 64, 128]

## Summary Table

| Parameter               | Value               |
| ----------------------- | ------------------- |
| BiLSTM Units            | 128                 |
| Attention Units         | 64                  |
| Dense Layer 1 Units     | 64                  |
| Dense Layer 2 Units     | 32                  |
| Learning Rate           | 0.001               |
| Batch Size              | 32                  |
| Epochs                  | 100                 |
| Dropout (BiLSTM)        | 0.2                 |
| Dropout (Dense 1)       | 0.3                 |
| Dropout (Dense 2)       | 0.2                 |
| Early Stopping Patience | 15                  |
| K-Fold CV               | 5                   |
| Selected Features (K)   | 20                  |
| Validation Split        | 0.2                 |
| Optimizer               | Adam                |
| Loss Function           | Binary Crossentropy |

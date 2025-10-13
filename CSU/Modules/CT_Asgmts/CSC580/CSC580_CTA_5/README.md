# Toxicology Hyperparameter Tuning - CSC580 CTA5

**Improving Accuracy of Tox21 Deep Learning Model through Systematic Hyperparameter Optimization**

## Overview

This project implements a comprehensive hyperparameter tuning system for the Tox21 toxicology prediction neural network. The program:

1. Loads the Tox21 dataset (single task - NR-AR)
2. Trains a Random Forest baseline for comparison
3. Implements TensorFlow fully connected neural network
4. Performs systematic hyperparameter search with nested loops
5. Runs multiple trials per configuration to handle random seed sensitivity
6. Tracks, analyzes, and visualizes best performing configurations

## Requirements

- Python 3.8+
- TensorFlow 2.x (with 1.x compatibility mode)
- DeepChem 2.7.1
- scikit-learn, matplotlib, pandas, numpy

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
./setup_environment.sh

# Activate the virtual environment
source tox_env/bin/activate
```

### 2. Run the Program

```bash
python3 toxicology_hyperparameter_tuning.py
```

**Note:** The default configuration tests 10 combinations × 3 trials = 30 training runs.
This takes approximately 30-60 minutes.

### 3. Adjust Configuration

To test more combinations, edit line 254 in the script:

```python
max_combinations = 10  # Change this to 50, 100, etc.
n_trials = 3           # Number of trials per config
```

## Program Structure

### Step 1: Load Data
```python
# Loads Tox21 and extracts single task (NR-AR)
train_y = train_y[:, 0]  # Only first toxicity assay
```

### Step 2: Random Forest Baseline
```python
# Traditional ML baseline for comparison
sklearn_model = RandomForestClassifier(class_weight="balanced", n_estimators=50)
```

### Step 3: TensorFlow Evaluation Function
```python
def eval_tox21_hyperparams(n_hidden, n_layers, learning_rate, ...):
    # TensorFlow 1.x style graph construction
    # Trains network and returns validation accuracy
```

### Step 4: Hyperparameter Search
```python
# Nested loops test all combinations
for n_hidden in [50, 100, 150]:
    for n_layers in [1, 2]:
        for lr in [0.0005, 0.001, 0.002]:
            # ... more parameters ...
            # Run multiple trials and average results
```

## Hyperparameter Search Space

```python
{
    'n_hidden': [50, 100, 150],              # Hidden units per layer
    'n_layers': [1, 2],                      # Number of layers
    'learning_rate': [0.0005, 0.001, 0.002], # Adam learning rate
    'dropout_prob': [0.3, 0.5, 0.7],         # Dropout probability
    'n_epochs': [30, 45],                    # Training epochs
    'batch_size': [64, 100, 128],            # Mini-batch size
    'weight_positives': [True, False]        # Weight positive class
}
```

**Total possible combinations:** 3 × 2 × 3 × 3 × 2 × 3 × 2 = **648 combinations**

## Output Files

After running, the program generates:

1. **random_forest_baseline.txt** - Random Forest performance metrics
2. **hyperparameter_results.csv** - Detailed results for all combinations
3. **best_configurations.txt** - Top 10 performing configurations
4. **hyperparameter_comparison.png** - 4-panel visualization:
   - Top configurations with error bars
   - Impact of hidden units
   - Impact of learning rate  
   - Impact of dropout

## Key Features

### Multiple Trials per Configuration
```python
n_trials = 3  # Each config trained 3 times
# Results are averaged to handle random seed variance
mean_accuracy = np.mean(trial_scores)
std_accuracy = np.std(trial_scores)
```

### TensorFlow 1.x Compatibility
```python
# Uses TF 2.x with 1.x compatibility mode
tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1  # Access TF 1.x API
```

### Dropout Handling
```python
# Training: keep_prob = dropout_prob (e.g., 0.5)
feed_dict = {x: batch_X, y: batch_y, w: batch_w, keep_prob: 0.5}

# Prediction: keep_prob = 1.0 (no dropout)
valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
```

## Expected Results

### Random Forest Baseline
- Train Accuracy: ~0.88-0.92
- Valid Accuracy: ~0.85-0.90
- Test Accuracy: ~0.85-0.90

### Neural Network (Optimized)
- Mean Accuracy: ~0.88-0.93 (averaged over trials)
- Std Accuracy: ~0.01-0.03 (stability measure)
- Improvement: +2-5% over Random Forest

## Time Estimates

| Combinations | Trials | Total Runs | Estimated Time |
|--------------|--------|------------|----------------|
| 10 | 3 | 30 | 30-60 min |
| 50 | 3 | 150 | 2-4 hours |
| 100 | 3 | 300 | 5-8 hours |

## Troubleshooting

### SSL Certificate Error
Already handled with:
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### TensorFlow Version Issues
The program auto-detects TensorFlow version and enables compatibility mode.

### Out of Memory
Reduce `batch_size` or `n_hidden` in hyperparameter grid.

### Slow Training
- Reduce `n_epochs` (30 instead of 45)
- Reduce `max_combinations` 
- Increase `batch_size`

## Analysis Questions Answered

1. **Which hyperparameters matter most?**
   - Visualized in hyperparameter_comparison.png
   - Top configurations show optimal ranges

2. **How stable are the results?**
   - Standard deviation shows sensitivity to random seeds
   - Lower std = more reliable configuration

3. **Does deep learning beat Random Forest?**
   - Direct comparison in results
   - Quantified improvement percentage

4. **What's the best configuration?**
   - Ranked in best_configurations.txt
   - Top performer clearly identified

## Assignment Deliverables

✅ **Step 1:** Load data (single task)  
✅ **Step 2:** Random Forest baseline  
✅ **Step 3:** TensorFlow evaluation function  
✅ **Step 4:** Hyperparameter search with nested loops and multiple trials  
✅ **Analysis:** Detailed tracking and comparison of results

## Author

Tripti Vishwakarma  
CSC580 - Artificial Intelligence  
Critical Thinking Assignment 5  
October 12, 2025

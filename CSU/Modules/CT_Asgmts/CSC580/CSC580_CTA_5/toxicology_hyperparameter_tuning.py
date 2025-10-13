#!/usr/bin/env python3
"""
Toxicology Neural Network - Hyperparameter Tuning
CSC580 - Critical Thinking Assignment 5

This program improves the accuracy of the Tox21 deep learning model by:
1. Loading the Tox21 dataset (single task)
2. Training Random Forest baseline
3. Implementing TensorFlow fully connected network
4. Systematic hyperparameter search with multiple trials
5. Tracking and analyzing best performers

Author: Tripti Vishwakarma
Date: October 12, 2025
"""

# Fix SSL certificate issue
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
np.random.seed(456)
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from datetime import datetime
import os

# TensorFlow setup with v1 compatibility
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

if tf.__version__.startswith('2'):
    print("Using TensorFlow 2.x - enabling v1 compatibility mode")
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf = tf.compat.v1
    print("✓ TensorFlow 1.x compatibility enabled")
elif tf.__version__.startswith('1'):
    print("Using TensorFlow 1.x - native support")
else:
    raise Exception(f"Unsupported TensorFlow version: {tf.__version__}")


print("\n" + "="*70)
print(" " * 15 + "TOXICOLOGY HYPERPARAMETER TUNING")
print(" " * 22 + "CSC580 - CTA5")
print("="*70)


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[STEP 1/4] Loading Tox21 Dataset...")
print("-" * 70)

# Workaround for DeepChem 2.5.0 metadata bug
print("  Loading dataset with DeepChem 2.5.0...")
import tempfile
import shutil

# Use temporary directory to force fresh download
with tempfile.TemporaryDirectory() as tmpdir:
    tasks, datasets, transformers = dc.molnet.load_tox21(
        featurizer='ECFP',
        splitter='random', 
        reload=True,
        data_dir=tmpdir
    )
    train, valid, test = datasets
    
    # Cache the data in current directory
    print("  Caching dataset locally...")
    os.makedirs('data_cache', exist_ok=True)
    import pickle
    with open('data_cache/tox21_data.pkl', 'wb') as f:
        pickle.dump((train, valid, test), f)

print("✓ Dataset loaded and cached successfully!")
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Remove extra tasks - keep only first task
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

print(f"✓ Dataset loaded successfully")
print(f"  Training samples: {train_X.shape[0]:,}")
print(f"  Validation samples: {valid_X.shape[0]:,}")
print(f"  Test samples: {test_X.shape[0]:,}")
print(f"  Features: {train_X.shape[1]:,}")
print(f"  Task: Single toxicity assay (NR-AR)")


# ============================================================================
# STEP 2: RANDOM FOREST BASELINE
# ============================================================================

print("\n[STEP 2/4] Training Random Forest Baseline...")
print("-" * 70)

sklearn_model = RandomForestClassifier(
    class_weight="balanced", 
    n_estimators=50,
    random_state=456,
    n_jobs=-1
)

print("  About to fit model on train set...")
sklearn_model.fit(train_X, train_y)

train_y_pred = sklearn_model.predict(train_X)
valid_y_pred = sklearn_model.predict(valid_X)
test_y_pred = sklearn_model.predict(test_X)

rf_train_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
rf_valid_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
rf_test_score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)

print(f"  Weighted train Classification Accuracy: {rf_train_score:.4f}")
print(f"  Weighted valid Classification Accuracy: {rf_valid_score:.4f}")
print(f"  Weighted test  Classification Accuracy: {rf_test_score:.4f}")

# Save Random Forest results
with open('random_forest_baseline.txt', 'w') as f:
    f.write("Random Forest Baseline Results\n")
    f.write("="*50 + "\n")
    f.write(f"Train Accuracy: {rf_train_score:.4f}\n")
    f.write(f"Valid Accuracy: {rf_valid_score:.4f}\n")
    f.write(f"Test Accuracy:  {rf_test_score:.4f}\n")
print("  ✓ Results saved to random_forest_baseline.txt")


# ============================================================================
# STEP 3: TENSORFLOW HYPERPARAMETER EVALUATION FUNCTION
# ============================================================================

def eval_tox21_hyperparams(n_hidden=50, n_layers=1, learning_rate=.001,
                           dropout_prob=0.5, n_epochs=45, batch_size=100,
                           weight_positives=True, track_history=False):
    """
    Evaluate performance of Tox21 models with hyperparameters.
    Returns accuracy and optionally training history.
        dropout_prob: Dropout probability (keep_prob during training)
        n_epochs: Number of training epochs
        batch_size: Mini-batch size
        weight_positives: Whether to weight positive samples
    
    Returns:
        weighted_score: Validation accuracy
    """
    print("---------------------------------------------")
    print("Model hyperparameters")
    print("n_hidden = %d" % n_hidden)
    print("n_layers = %d" % n_layers)
    print("learning_rate = %f" % learning_rate)
    print("n_epochs = %d" % n_epochs)
    print("batch_size = %d" % batch_size)
    print("weight_positives = %s" % str(weight_positives))
    print("dropout_prob = %f" % dropout_prob)
    print("---------------------------------------------")
    
    d = 1024  # Input dimension
    graph = tf.Graph()
    
    with graph.as_default():
        # Load cached data to avoid metadata bug
        import pickle
        if os.path.exists('data_cache/tox21_data.pkl'):
            with open('data_cache/tox21_data.pkl', 'rb') as f:
                train, valid, test = pickle.load(f)
        else:
            # Fallback
            _, (train, valid, test), _ = dc.molnet.load_tox21()
        train_X, train_y, train_w = train.X, train.y, train.w
        valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
        test_X, test_y, test_w = test.X, test.y, test.w
        
        # Remove extra tasks
        train_y = train_y[:, 0]
        valid_y = valid_y[:, 0]
        test_y = test_y[:, 0]
        train_w = train_w[:, 0]
        valid_w = valid_w[:, 0]
        test_w = test_w[:, 0]
        
        # Generate tensorflow graph
        with tf.name_scope("placeholders"):
            x = tf.placeholder(tf.float32, (None, d))
            y = tf.placeholder(tf.float32, (None,))
            w = tf.placeholder(tf.float32, (None,))
            keep_prob = tf.placeholder(tf.float32)
        
        # Build hidden layers
        for layer in range(n_layers):
            with tf.name_scope("layer-%d" % layer):
                W = tf.Variable(tf.random_normal((d, n_hidden)))
                b = tf.Variable(tf.random_normal((n_hidden,)))
                x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
                # Apply dropout
                x_hidden = tf.nn.dropout(x_hidden, keep_prob)
        
        # Output layer
        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal((n_hidden, 1)))
            b = tf.Variable(tf.random_normal((1,)))
            y_logit = tf.matmul(x_hidden, W) + b
            # the sigmoid gives the class probability of 1
            y_one_prob = tf.sigmoid(y_logit)
            # Rounding P(y=1) will give the correct prediction.
            y_pred = tf.round(y_one_prob)
        
        # Loss function
        with tf.name_scope("loss"):
            # Compute the cross-entropy term for each datapoint
            y_expand = tf.expand_dims(y, 1)
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
            # Multiply by weights
            if weight_positives:
                w_expand = tf.expand_dims(w, 1)
                entropy = w_expand * entropy
            # Sum all contributions
            l = tf.reduce_sum(entropy)
        
        # Optimizer
        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
        
        # TensorBoard summaries
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", l)
            merged = tf.summary.merge_all()
        
        hyperparam_str = "d-%d-hidden-%d-lr-%f-n_epochs-%d-batch_size-%d-weight_pos-%s" % (
            d, n_hidden, learning_rate, n_epochs, batch_size, str(weight_positives))
        train_writer = tf.summary.FileWriter('/tmp/fcnet-func-' + hyperparam_str,
                                             tf.get_default_graph())
        
        N = train_X.shape[0]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            for epoch in range(n_epochs):
                pos = 0
                while pos < N:
                    batch_X = train_X[pos:pos+batch_size]
                    batch_y = train_y[pos:pos+batch_size]
                    batch_w = train_w[pos:pos+batch_size]
                    feed_dict = {x: batch_X, y: batch_y, w: batch_w, keep_prob: dropout_prob}
                    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
                    
                    if step % 10 == 0:  # Print every 10 steps
                        print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
                    train_writer.add_summary(summary, step)
                    
                    step += 1
                    pos += batch_size
            
            # Make Predictions (set keep_prob to 1.0 for predictions)
            valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
        
        weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
        print("Valid Weighted Classification Accuracy: %f" % weighted_score)
        
    return weighted_score


# ============================================================================
# STEP 4: HYPERPARAMETER SEARCH WITH MULTIPLE TRIALS
# ============================================================================

print("\n[STEP 3/4] Hyperparameter Search Configuration...")
print("-" * 70)

# Define hyperparameter search space
hyperparameter_grid = {
    'n_hidden': [50, 100, 150],
    'n_layers': [1, 2],
    'learning_rate': [0.0005, 0.001, 0.002],
    'dropout_prob': [0.3, 0.5, 0.7],
    'n_epochs': [30, 45],
    'batch_size': [64, 100, 128],
    'weight_positives': [True, False]
}

print("Hyperparameter Search Space:")
for param, values in hyperparameter_grid.items():
    print(f"  {param}: {values}")

# Calculate total combinations
total_combinations = 1
for values in hyperparameter_grid.values():
    total_combinations *= len(values)

print(f"\nTotal possible combinations: {total_combinations:,}")

# For demonstration, test a subset
max_combinations = 10  # Change this to test more combinations
n_trials = 3  # Number of trials per configuration to average

print(f"Testing: {max_combinations} combinations")
print(f"Trials per combination: {n_trials}")
print(f"Total training runs: {max_combinations * n_trials}")

print("\n[STEP 4/4] Running Hyperparameter Search...")
print("-" * 70)
print("⚠️  This may take 1-2 hours depending on configurations tested")
print()

# Storage for results
results = []
combination_count = 0

# Nested loops for hyperparameter search
for n_hidden in hyperparameter_grid['n_hidden']:
    for n_layers in hyperparameter_grid['n_layers']:
        for lr in hyperparameter_grid['learning_rate']:
            for dropout in hyperparameter_grid['dropout_prob']:
                for epochs in hyperparameter_grid['n_epochs']:
                    for batch in hyperparameter_grid['batch_size']:
                        for weight_pos in hyperparameter_grid['weight_positives']:
                            
                            if combination_count >= max_combinations:
                                print(f"\n✓ Reached max combinations limit ({max_combinations})")
                                break
                            
                            combination_count += 1
                            
                            print(f"\n{'='*70}")
                            print(f"COMBINATION {combination_count}/{max_combinations}")
                            print(f"{'='*70}")
                            print(f"Config: n_hidden={n_hidden}, n_layers={n_layers}, lr={lr},")
                            print(f"        dropout={dropout}, epochs={epochs}, batch={batch}, weight_pos={weight_pos}")
                            
                            # Run multiple trials
                            trial_scores = []
                            for trial in range(n_trials):
                                print(f"\n--- Trial {trial+1}/{n_trials} ---")
                                
                                try:
                                    score = eval_tox21_hyperparams(
                                        n_hidden=n_hidden,
                                        n_layers=n_layers,
                                        learning_rate=lr,
                                        dropout_prob=dropout,
                                        n_epochs=epochs,
                                        batch_size=batch,
                                        weight_positives=weight_pos
                                    )
                                    trial_scores.append(score)
                                except Exception as e:
                                    print(f"  ✗ Error in trial {trial+1}: {str(e)[:100]}")
                                    continue
                            
                            if trial_scores:
                                mean_score = np.mean(trial_scores)
                                std_score = np.std(trial_scores)
                                
                                result_entry = {
                                    'combination': combination_count,
                                    'n_hidden': n_hidden,
                                    'n_layers': n_layers,
                                    'learning_rate': lr,
                                    'dropout_prob': dropout,
                                    'n_epochs': epochs,
                                    'batch_size': batch,
                                    'weight_positives': weight_pos,
                                    'trial_scores': trial_scores,
                                    'mean_accuracy': mean_score,
                                    'std_accuracy': std_score,
                                    'n_trials': len(trial_scores)
                                }
                                
                                results.append(result_entry)
                                
                                print(f"\n✓ AVERAGED RESULTS:")
                                print(f"  Mean Accuracy: {mean_score:.4f} ± {std_score:.4f}")
                                print(f"  Individual trials: {[f'{s:.4f}' for s in trial_scores]}")
                        
                        if combination_count >= max_combinations:
                            break
                    if combination_count >= max_combinations:
                        break
                if combination_count >= max_combinations:
                    break
            if combination_count >= max_combinations:
                break
        if combination_count >= max_combinations:
            break
    if combination_count >= max_combinations:
        break


# ============================================================================
# RESULTS ANALYSIS AND VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print(" " * 20 + "RESULTS ANALYSIS")
print("="*70)

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mean_accuracy', ascending=False)

# Save detailed results
results_df.to_csv('hyperparameter_results.csv', index=False)
print("\n✓ Detailed results saved to: hyperparameter_results.csv")

# Display top 10 configurations
top_k = min(10, len(results_df))
print(f"\nTOP {top_k} HYPERPARAMETER CONFIGURATIONS:")
print("="*70)

for idx, row in results_df.head(top_k).iterrows():
    rank = list(results_df.index).index(idx) + 1
    print(f"\nRank #{rank}:")
    print(f"  n_hidden={int(row['n_hidden'])}, n_layers={int(row['n_layers'])}, lr={row['learning_rate']:.4f}")
    print(f"  dropout={row['dropout_prob']}, epochs={int(row['n_epochs'])}, batch={int(row['batch_size'])}")
    print(f"  weight_positives={row['weight_positives']}")
    print(f"  → Mean Accuracy: {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}")

# Save top configurations
with open('best_configurations.txt', 'w') as f:
    f.write("TOP 10 HYPERPARAMETER CONFIGURATIONS\n")
    f.write("="*70 + "\n\n")
    for idx, row in results_df.head(top_k).iterrows():
        rank = list(results_df.index).index(idx) + 1
        f.write(f"Rank #{rank}:\n")
        f.write(f"  n_hidden={int(row['n_hidden'])}, n_layers={int(row['n_layers'])}, ")
        f.write(f"learning_rate={row['learning_rate']:.4f}\n")
        f.write(f"  dropout_prob={row['dropout_prob']}, n_epochs={int(row['n_epochs'])}, ")
        f.write(f"batch_size={int(row['batch_size'])}\n")
        f.write(f"  weight_positives={row['weight_positives']}\n")
        f.write(f"  Mean Accuracy: {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}\n\n")

print("\n✓ Top configurations saved to: best_configurations.txt")


# ============================================================================
# STEP 5: EVALUATE BEST MODEL ON TEST SET
# ============================================================================

print("\n[STEP 4/4] Evaluating Best Model on Test Set...")
print("-" * 70)

# Get best hyperparameters
best_config = results_df.iloc[0]
print("Best Configuration:")
print(f"  n_hidden: {int(best_config['n_hidden'])}")
print(f"  n_layers: {int(best_config['n_layers'])}")
print(f"  learning_rate: {best_config['learning_rate']}")
print(f"  dropout_prob: {best_config['dropout_prob']}")
print(f"  n_epochs: {int(best_config['n_epochs'])}")
print(f"  batch_size: {int(best_config['batch_size'])}")
print(f"  weight_positives: {best_config['weight_positives']}")

print("\nRetraining best model for final evaluation...")

# Final evaluation function with train, valid, and test metrics
def eval_final_model(n_hidden=50, n_layers=1, learning_rate=.001,
                     dropout_prob=0.5, n_epochs=45, batch_size=100,
                     weight_positives=True):
    
    d = 1024
    graph = tf.Graph()
    
    with graph.as_default():
        # Load cached data
        import pickle
        if os.path.exists('data_cache/tox21_data.pkl'):
            with open('data_cache/tox21_data.pkl', 'rb') as f:
                train, valid, test = pickle.load(f)
        else:
            _, (train, valid, test), _ = dc.molnet.load_tox21()
        
        train_X, train_y, train_w = train.X, train.y, train.w
        valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
        test_X, test_y, test_w = test.X, test.y, test.w
        
        # Remove extra tasks
        train_y = train_y[:, 0]
        valid_y = valid_y[:, 0]
        test_y = test_y[:, 0]
        train_w = train_w[:, 0]
        valid_w = valid_w[:, 0]
        test_w = test_w[:, 0]
        
        # Build graph
        with tf.name_scope("placeholders"):
            x = tf.placeholder(tf.float32, (None, d))
            y = tf.placeholder(tf.float32, (None,))
            w = tf.placeholder(tf.float32, (None,))
            keep_prob = tf.placeholder(tf.float32)
        
        for layer in range(n_layers):
            with tf.name_scope("layer-%d" % layer):
                W = tf.Variable(tf.random_normal((d, n_hidden)))
                b = tf.Variable(tf.random_normal((n_hidden,)))
                x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
                x_hidden = tf.nn.dropout(x_hidden, keep_prob)
        
        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal((n_hidden, 1)))
            b = tf.Variable(tf.random_normal((1,)))
            y_logit = tf.matmul(x_hidden, W) + b
            y_one_prob = tf.sigmoid(y_logit)
            y_pred = tf.round(y_one_prob)
        
        with tf.name_scope("loss"):
            y_expand = tf.expand_dims(y, 1)
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
            if weight_positives:
                w_expand = tf.expand_dims(w, 1)
                entropy = w_expand * entropy
            l = tf.reduce_sum(entropy)
        
        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
        
        N = train_X.shape[0]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            
            # Training loop (suppressed output)
            for epoch in range(n_epochs):
                pos = 0
                while pos < N:
                    batch_X = train_X[pos:pos+batch_size]
                    batch_y = train_y[pos:pos+batch_size]
                    batch_w = train_w[pos:pos+batch_size]
                    feed_dict = {x: batch_X, y: batch_y, w: batch_w, keep_prob: dropout_prob}
                    _, loss = sess.run([train_op, l], feed_dict=feed_dict)
                    
                    if step % 50 == 0:
                        print(f"  Training... epoch {epoch}/{n_epochs}, step {step}, loss: {loss:.2f}")
                    
                    step += 1
                    pos += batch_size
            
            # Evaluate on all sets
            train_y_pred = sess.run(y_pred, feed_dict={x: train_X, keep_prob: 1.0})
            valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
            test_y_pred = sess.run(y_pred, feed_dict={x: test_X, keep_prob: 1.0})
        
        train_acc = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
        valid_acc = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
        test_acc = accuracy_score(test_y, test_y_pred, sample_weight=test_w)
        
    return train_acc, valid_acc, test_acc

# Train final model with best hyperparameters
final_train_acc, final_valid_acc, final_test_acc = eval_final_model(
    n_hidden=int(best_config['n_hidden']),
    n_layers=int(best_config['n_layers']),
    learning_rate=best_config['learning_rate'],
    dropout_prob=best_config['dropout_prob'],
    n_epochs=int(best_config['n_epochs']),
    batch_size=int(best_config['batch_size']),
    weight_positives=best_config['weight_positives']
)

print("\n" + "="*70)
print("FINAL MODEL RESULTS (Best Hyperparameters)")
print("="*70)
print(f"Train Accuracy:      {final_train_acc:.4f}")
print(f"Validation Accuracy: {final_valid_acc:.4f}")
print(f"Test Accuracy:       {final_test_acc:.4f}")

# Save final results
with open('final_model_results.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("FINAL MODEL EVALUATION\n")
    f.write("="*70 + "\n\n")
    f.write("Best Hyperparameters:\n")
    f.write(f"  n_hidden: {int(best_config['n_hidden'])}\n")
    f.write(f"  n_layers: {int(best_config['n_layers'])}\n")
    f.write(f"  learning_rate: {best_config['learning_rate']}\n")
    f.write(f"  dropout_prob: {best_config['dropout_prob']}\n")
    f.write(f"  n_epochs: {int(best_config['n_epochs'])}\n")
    f.write(f"  batch_size: {int(best_config['batch_size'])}\n")
    f.write(f"  weight_positives: {best_config['weight_positives']}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"  Train Accuracy:      {final_train_acc:.4f}\n")
    f.write(f"  Validation Accuracy: {final_valid_acc:.4f}\n")
    f.write(f"  Test Accuracy:       {final_test_acc:.4f}\n\n")
    f.write("Baseline Comparison:\n")
    f.write(f"  Random Forest (valid): {rf_valid_score:.4f}\n")
    f.write(f"  Random Forest (test):  {rf_test_score:.4f}\n")
    f.write(f"  Neural Network (test): {final_test_acc:.4f}\n")
    f.write(f"  Improvement over RF:   {(final_test_acc - rf_test_score) * 100:+.2f}%\n")

print("  ✓ Results saved to: final_model_results.txt")

# Comparison with Random Forest
best_nn_accuracy = results_df.iloc[0]['mean_accuracy']
improvement = (best_nn_accuracy - rf_valid_score) * 100

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"Random Forest (baseline):     {rf_valid_score:.4f}")
print(f"Best Neural Network:          {best_nn_accuracy:.4f}")
print(f"Improvement:                  {improvement:+.2f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[VISUALIZATION] Generating plots...")

# Plot 1: Top configurations comparison
if len(results_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top configurations with error bars
    top_results = results_df.head(min(10, len(results_df)))
    x = np.arange(len(top_results))
    
    axes[0, 0].bar(x, top_results['mean_accuracy'], yerr=top_results['std_accuracy'], 
                   capsize=5, alpha=0.7, color='steelblue')
    axes[0, 0].axhline(y=rf_valid_score, color='red', linestyle='--', label='Random Forest', linewidth=2)
    axes[0, 0].set_title('Top Configurations - Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Configuration Rank')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Hyperparameter impact: n_hidden
    grouped = results_df.groupby('n_hidden')['mean_accuracy'].mean()
    axes[0, 1].bar(grouped.index.astype(str), grouped.values, alpha=0.7, color='coral')
    axes[0, 1].set_title('Impact of Hidden Units', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Hidden Units')
    axes[0, 1].set_ylabel('Mean Accuracy')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Hyperparameter impact: learning_rate
    grouped = results_df.groupby('learning_rate')['mean_accuracy'].mean()
    axes[1, 0].bar(grouped.index.astype(str), grouped.values, alpha=0.7, color='green')
    axes[1, 0].set_title('Impact of Learning Rate', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Learning Rate')
    axes[1, 0].set_ylabel('Mean Accuracy')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Hyperparameter impact: dropout
    grouped = results_df.groupby('dropout_prob')['mean_accuracy'].mean()
    axes[1, 1].bar(grouped.index.astype(str), grouped.values, alpha=0.7, color='purple')
    axes[1, 1].set_title('Impact of Dropout Probability', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Dropout Probability')
    axes[1, 1].set_ylabel('Mean Accuracy')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: hyperparameter_comparison.png")
    print("  📊 Displaying graph... (close window to continue)")
    plt.show()
    plt.close()

# Plot 2: Model Performance Comparison
print("\n[VISUALIZATION] Generating performance comparison plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Best configurations comparison
top_10 = results_df.head(10)
x_pos = np.arange(len(top_10))

axes[0, 0].barh(x_pos, top_10['mean_accuracy'], xerr=top_10['std_accuracy'], 
               capsize=5, alpha=0.7, color='#2E86AB')
axes[0, 0].axvline(x=rf_valid_score, color='red', linestyle='--', 
                   label=f'Random Forest ({rf_valid_score:.4f})', linewidth=2)
axes[0, 0].set_yticks(x_pos)
axes[0, 0].set_yticklabels([f"Config {i+1}" for i in range(len(top_10))])
axes[0, 0].set_xlabel('Validation Accuracy', fontsize=12)
axes[0, 0].set_title('Top 10 Configurations vs Baseline', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='x')
axes[0, 0].invert_yaxis()

# Batch size impact
batch_grouped = results_df.groupby('batch_size').agg({
    'mean_accuracy': ['mean', 'std']
}).reset_index()
batch_grouped.columns = ['batch_size', 'mean', 'std']

axes[0, 1].bar(batch_grouped['batch_size'].astype(str), batch_grouped['mean'], 
              yerr=batch_grouped['std'], capsize=5, alpha=0.7, color='#A23B72')
axes[0, 1].axhline(y=rf_valid_score, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Batch Size', fontsize=12)
axes[0, 1].set_ylabel('Mean Accuracy', fontsize=12)
axes[0, 1].set_title('Impact of Batch Size', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Epochs impact
epoch_grouped = results_df.groupby('n_epochs').agg({
    'mean_accuracy': ['mean', 'std']
}).reset_index()
epoch_grouped.columns = ['n_epochs', 'mean', 'std']

axes[1, 0].bar(epoch_grouped['n_epochs'].astype(str), epoch_grouped['mean'], 
              yerr=epoch_grouped['std'], capsize=5, alpha=0.7, color='#F18F01')
axes[1, 0].axhline(y=rf_valid_score, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Number of Epochs', fontsize=12)
axes[1, 0].set_ylabel('Mean Accuracy', fontsize=12)
axes[1, 0].set_title('Impact of Training Epochs', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Weight positives impact
weight_grouped = results_df.groupby('weight_positives').agg({
    'mean_accuracy': ['mean', 'std']
}).reset_index()
weight_grouped.columns = ['weight_positives', 'mean', 'std']
weight_labels = ['No Weighting', 'Weight Positives']

axes[1, 1].bar(weight_labels, weight_grouped['mean'], 
              yerr=weight_grouped['std'], capsize=5, alpha=0.7, 
              color=['#C73E1D', '#6A994E'])
axes[1, 1].axhline(y=rf_valid_score, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_ylabel('Mean Accuracy', fontsize=12)
axes[1, 1].set_title('Impact of Sample Weighting', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: model_performance_analysis.png")
print("  📊 Displaying graph... (close window to continue)")
plt.show()
plt.close()

# Plot 3: Hyperparameter Heatmap
print("\n[VISUALIZATION] Generating hyperparameter heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Learning rate vs Hidden units
pivot_lr_hidden = results_df.pivot_table(
    values='mean_accuracy', 
    index='learning_rate', 
    columns='n_hidden', 
    aggfunc='mean'
)

im1 = axes[0].imshow(pivot_lr_hidden.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.8)
axes[0].set_xticks(range(len(pivot_lr_hidden.columns)))
axes[0].set_yticks(range(len(pivot_lr_hidden.index)))
axes[0].set_xticklabels(pivot_lr_hidden.columns)
axes[0].set_yticklabels([f"{lr:.4f}" for lr in pivot_lr_hidden.index])
axes[0].set_xlabel('Hidden Units', fontsize=12)
axes[0].set_ylabel('Learning Rate', fontsize=12)
axes[0].set_title('Learning Rate vs Hidden Units', fontsize=14, fontweight='bold')

for i in range(len(pivot_lr_hidden.index)):
    for j in range(len(pivot_lr_hidden.columns)):
        if not np.isnan(pivot_lr_hidden.iloc[i, j]):
            axes[0].text(j, i, f'{pivot_lr_hidden.iloc[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im1, ax=axes[0], label='Accuracy')

# Dropout vs Batch size
pivot_dropout_batch = results_df.pivot_table(
    values='mean_accuracy', 
    index='dropout_prob', 
    columns='batch_size', 
    aggfunc='mean'
)

im2 = axes[1].imshow(pivot_dropout_batch.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.8)
axes[1].set_xticks(range(len(pivot_dropout_batch.columns)))
axes[1].set_yticks(range(len(pivot_dropout_batch.index)))
axes[1].set_xticklabels(pivot_dropout_batch.columns)
axes[1].set_yticklabels(pivot_dropout_batch.index)
axes[1].set_xlabel('Batch Size', fontsize=12)
axes[1].set_ylabel('Dropout Probability', fontsize=12)
axes[1].set_title('Dropout vs Batch Size', fontsize=14, fontweight='bold')

for i in range(len(pivot_dropout_batch.index)):
    for j in range(len(pivot_dropout_batch.columns)):
        if not np.isnan(pivot_dropout_batch.iloc[i, j]):
            axes[1].text(j, i, f'{pivot_dropout_batch.iloc[i, j]:.3f}',
                        ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im2, ax=axes[1], label='Accuracy')

plt.tight_layout()
plt.savefig('hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: hyperparameter_heatmap.png")
print("  📊 Displaying graph... (close window to continue)")
plt.show()
plt.close()

# Plot 4: Summary Statistics
print("\n[VISUALIZATION] Generating summary statistics...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Overall distribution
ax1 = fig.add_subplot(gs[0, :])
ax1.hist(results_df['mean_accuracy'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(rf_valid_score, color='red', linestyle='--', linewidth=2, label='Random Forest')
ax1.axvline(best_nn_accuracy, color='green', linestyle='--', linewidth=2, label='Best NN')
ax1.set_xlabel('Validation Accuracy', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Model Performance', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Box plots for each hyperparameter
hyperparams_to_plot = ['n_hidden', 'learning_rate', 'dropout_prob']
for idx, param in enumerate(hyperparams_to_plot):
    ax = fig.add_subplot(gs[1, idx])
    grouped_data = [results_df[results_df[param] == val]['mean_accuracy'].values 
                    for val in sorted(results_df[param].unique())]
    ax.boxplot(grouped_data, labels=[str(v) for v in sorted(results_df[param].unique())])
    ax.axhline(rf_valid_score, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel(param.replace('_', ' ').title(), fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

# Scatter plots
ax4 = fig.add_subplot(gs[2, 0])
for weight_val in results_df['weight_positives'].unique():
    subset = results_df[results_df['weight_positives'] == weight_val]
    label = 'Weighted' if weight_val else 'Unweighted'
    ax4.scatter(subset['n_hidden'], subset['mean_accuracy'], 
               alpha=0.6, s=100, label=label)
ax4.set_xlabel('Hidden Units', fontsize=10)
ax4.set_ylabel('Accuracy', fontsize=10)
ax4.set_title('Hidden Units vs Accuracy', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[2, 1])
for weight_val in results_df['weight_positives'].unique():
    subset = results_df[results_df['weight_positives'] == weight_val]
    label = 'Weighted' if weight_val else 'Unweighted'
    ax5.scatter(subset['learning_rate'], subset['mean_accuracy'], 
               alpha=0.6, s=100, label=label)
ax5.set_xlabel('Learning Rate', fontsize=10)
ax5.set_ylabel('Accuracy', fontsize=10)
ax5.set_title('Learning Rate vs Accuracy', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[2, 2])
for weight_val in results_df['weight_positives'].unique():
    subset = results_df[results_df['weight_positives'] == weight_val]
    label = 'Weighted' if weight_val else 'Unweighted'
    ax6.scatter(subset['dropout_prob'], subset['mean_accuracy'], 
               alpha=0.6, s=100, label=label)
ax6.set_xlabel('Dropout Probability', fontsize=10)
ax6.set_ylabel('Accuracy', fontsize=10)
ax6.set_title('Dropout vs Accuracy', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.savefig('hyperparameter_summary.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: hyperparameter_summary.png")
print("  📊 Displaying graph... (close window to continue)")
plt.show()
plt.close()

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

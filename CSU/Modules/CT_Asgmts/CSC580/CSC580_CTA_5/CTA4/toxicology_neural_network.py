#!/usr/bin/env python3
"""
Toxicology Testing Neural Network
CSC580 - Critical Thinking Assignment 4

This script implements a neural network to predict human toxicity reactions
to chemical compounds using the Tox21 dataset following 9 specific steps.

Author: Tripti Vishwakarma
Date: 2025-10-05
"""

# Fix SSL certificate issue on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Step 1: Load the Tox21 Dataset
import numpy as np
np.random.seed(456)
import tensorflow as tf
if hasattr(tf, 'set_random_seed'):
    tf.set_random_seed(456)  # TensorFlow 1.x
else:
    tf.random.set_seed(456)  # TensorFlow 2.x
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from datetime import datetime

class ToxicologyPredictor:
    """
    Neural network model for predicting chemical compound toxicity.
    
    The Tox21 dataset contains information about 12 different toxicity assays:
    - NR-AR: Nuclear receptor - Androgen receptor
    - NR-AR-LBD: Nuclear receptor - Androgen receptor ligand binding domain
    - NR-AhR: Nuclear receptor - Aryl hydrocarbon receptor
    - NR-Aromatase: Nuclear receptor - Aromatase
    - NR-ER: Nuclear receptor - Estrogen receptor
    - NR-ER-LBD: Nuclear receptor - Estrogen receptor ligand binding domain
    - NR-PPAR-gamma: Nuclear receptor - Peroxisome proliferator-activated receptor gamma
    - SR-ARE: Stress response - Antioxidant response element
    - SR-ATAD5: Stress response - ATAD5
    - SR-HSE: Stress response - Heat shock factor response element
    - SR-MMP: Stress response - Mitochondrial membrane potential
    - SR-p53: Stress response - Tumor suppressor p53
    """
    
    def __init__(self, input_dim, num_tasks=12):
        """
        Initialize the toxicology predictor.
        
        Args:
            input_dim: Number of input features (molecular descriptors)
            num_tasks: Number of toxicity prediction tasks (default: 12)
        """
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.model = None
        self.history = None
        self.task_names = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
    def build_model(self, hidden_layers=[1000, 500, 250], dropout_rate=0.3):
        """
        Build a deep neural network for multi-task toxicity prediction.
        
        Args:
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        print("\n" + "="*70)
        print("BUILDING NEURAL NETWORK ARCHITECTURE")
        print("="*70)
        
        # Input layer
        inputs = tf.keras.Input(shape=(self.input_dim,), name='molecular_features')
        
        # Hidden layers with batch normalization and dropout
        x = inputs
        for i, units in enumerate(hidden_layers):
            x = tf.keras.layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name=f'hidden_layer_{i+1}'
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Output layer for multi-task classification
        outputs = tf.keras.layers.Dense(
            self.num_tasks,
            activation='sigmoid',
            name='toxicity_predictions'
        )(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ToxicologyPredictor')
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print("\nModel Architecture:")
        print(f"  Input Dimension: {self.input_dim} molecular features")
        print(f"  Hidden Layers: {hidden_layers}")
        print(f"  Dropout Rate: {dropout_rate}")
        print(f"  Output Tasks: {self.num_tasks} toxicity assays")
        print(f"  Total Parameters: {self.model.count_params():,}")
        print("\nModel Summary:")
        self.model.summary()
        
    def train(self, train_X, train_y, train_w, valid_X, valid_y, valid_w, 
              epochs=50, batch_size=128, verbose=1):
        """
        Train the neural network on toxicity data.
        
        Args:
            train_X: Training features
            train_y: Training labels
            train_w: Training sample weights
            valid_X: Validation features
            valid_y: Validation labels
            valid_w: Validation sample weights
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
        """
        print("\n" + "="*70)
        print("TRAINING NEURAL NETWORK")
        print("="*70)
        print(f"\nTraining Configuration:")
        print(f"  Training Samples: {len(train_X):,}")
        print(f"  Validation Samples: {len(valid_X):,}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Learning Rate: 0.001")
        
        # Create TensorBoard log directory
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Callbacks for training
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,  # Disable to avoid crashes
                write_graph=True,  # Enable model graph
                write_images=False,  # Disable to avoid crashes
                update_freq='epoch',
                profile_batch=0  # Disable profiling to avoid crashes
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print(f"\nTensorBoard logs directory: {log_dir}")
        print(f"To view TensorBoard, run: tensorboard --logdir=logs/fit")
        
        # Train the model
        # Average sample weights across tasks for Keras compatibility
        train_w_avg = np.mean(train_w, axis=1)
        valid_w_avg = np.mean(valid_w, axis=1)
        
        start_time = datetime.now()
        self.history = self.model.fit(
            train_X, train_y,
            sample_weight=train_w_avg,
            validation_data=(valid_X, valid_y, valid_w_avg),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
    def evaluate(self, test_X, test_y, test_w):
        """
        Evaluate the model on test data.
        
        Args:
            test_X: Test features
            test_y: Test labels
            test_w: Test sample weights
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("\n" + "="*70)
        print("EVALUATING MODEL PERFORMANCE")
        print("="*70)
        
        # Get predictions
        predictions = self.model.predict(test_X, verbose=0)
        
        # Calculate metrics for each task
        results = {
            'task_name': [],
            'accuracy': [],
            'roc_auc': [],
            'samples': []
        }
        
        print(f"\nTest Set: {len(test_X):,} samples\n")
        print(f"{'Task Name':<25} {'Accuracy':<12} {'ROC-AUC':<12} {'Samples':<10}")
        print("-" * 70)
        
        for task_idx in range(self.num_tasks):
            # Get valid samples (non-missing labels)
            valid_indices = test_w[:, task_idx] != 0
            
            if valid_indices.sum() > 0:
                y_true = test_y[valid_indices, task_idx]
                y_pred = predictions[valid_indices, task_idx]
                y_pred_binary = (y_pred > 0.5).astype(int)
                
                # Calculate metrics
                acc = accuracy_score(y_true, y_pred_binary)
                try:
                    auc = roc_auc_score(y_true, y_pred)
                except:
                    auc = 0.0  # Handle cases with only one class
                
                results['task_name'].append(self.task_names[task_idx])
                results['accuracy'].append(acc)
                results['roc_auc'].append(auc)
                results['samples'].append(valid_indices.sum())
                
                print(f"{self.task_names[task_idx]:<25} {acc:<12.4f} {auc:<12.4f} {valid_indices.sum():<10}")
        
        # Calculate average metrics
        avg_accuracy = np.mean(results['accuracy'])
        avg_auc = np.mean([auc for auc in results['roc_auc'] if auc > 0])
        
        print("-" * 70)
        print(f"{'AVERAGE':<25} {avg_accuracy:<12.4f} {avg_auc:<12.4f}")
        print("\n" + "="*70)
        
        return results
    
    def predict_compound(self, features, threshold=0.5):
        """
        Predict toxicity for a single compound.
        
        Args:
            features: Molecular features of the compound
            threshold: Classification threshold
            
        Returns:
            Dictionary with predictions for each assay
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        predictions = self.model.predict(features, verbose=0)[0]
        
        results = {}
        print("\n" + "="*70)
        print("COMPOUND TOXICITY PREDICTION")
        print("="*70)
        print(f"\n{'Assay Name':<25} {'Probability':<15} {'Prediction':<15}")
        print("-" * 70)
        
        for i, task_name in enumerate(self.task_names):
            prob = predictions[i]
            pred = "TOXIC" if prob > threshold else "NON-TOXIC"
            results[task_name] = {'probability': prob, 'prediction': pred}
            print(f"{task_name:<25} {prob:<15.4f} {pred:<15}")
        
        print("="*70)
        return results
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training and validation metrics over epochs.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot loss
        epochs = range(1, len(self.history.history['loss']) + 1)
        axes[0].plot(epochs, self.history.history['loss'], 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=4)
        axes[0].plot(epochs, self.history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=4)
        axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
        axes[0].legend(fontsize=11, loc='upper right')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Highlight best epoch
        best_epoch = np.argmin(self.history.history['val_loss']) + 1
        best_val_loss = np.min(self.history.history['val_loss'])
        axes[0].axvline(x=best_epoch, color='g', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch: {best_epoch}')
        axes[0].plot(best_epoch, best_val_loss, 'g*', markersize=15, label=f'Min Val Loss: {best_val_loss:.4f}')
        axes[0].legend(fontsize=10, loc='upper right')
        
        # Plot accuracy
        axes[1].plot(epochs, self.history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2.5, marker='o', markersize=4)
        axes[1].plot(epochs, self.history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2.5, marker='s', markersize=4)
        axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=11, loc='lower right')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_ylim([0, 1])
        
        # Plot AUC
        axes[2].plot(epochs, self.history.history['auc'], 'b-', label='Training AUC', linewidth=2.5, marker='o', markersize=4)
        axes[2].plot(epochs, self.history.history['val_auc'], 'r-', label='Validation AUC', linewidth=2.5, marker='s', markersize=4)
        axes[2].set_title('Model AUC Over Epochs', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Area Under ROC Curve', fontsize=12)
        axes[2].legend(fontsize=11, loc='lower right')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].set_ylim([0.5, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to: {save_path}")
        plt.show()
        plt.close()
    
    def plot_loss_curve(self, save_path='loss_curve.png'):
        """
        Plot detailed loss curve (separate from training history).
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        plt.figure(figsize=(12, 7))
        epochs = range(1, len(self.history.history['loss']) + 1)
        
        # Plot loss curves with larger, more prominent styling
        plt.plot(epochs, self.history.history['loss'], 'b-', label='Training Loss', 
                linewidth=3, marker='o', markersize=6, markerfacecolor='blue', markeredgewidth=1.5, markeredgecolor='darkblue')
        plt.plot(epochs, self.history.history['val_loss'], 'r-', label='Validation Loss', 
                linewidth=3, marker='s', markersize=6, markerfacecolor='red', markeredgewidth=1.5, markeredgecolor='darkred')
        
        # Highlight best epoch
        best_epoch = np.argmin(self.history.history['val_loss']) + 1
        best_val_loss = np.min(self.history.history['val_loss'])
        plt.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7)
        plt.plot(best_epoch, best_val_loss, 'g*', markersize=20, label=f'Best Epoch: {best_epoch}')
        
        # Add annotations
        plt.annotate(f'Min Val Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}',
                    xy=(best_epoch, best_val_loss),
                    xytext=(best_epoch + 3, best_val_loss + 0.3),
                    fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2))
        
        # Formatting
        plt.title('Toxicology Neural Network - Loss Curve', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Binary Cross-Entropy Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
        plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve plot saved to: {save_path}")
        plt.show()
        plt.close()
    
    def plot_task_performance(self, results, save_path='task_performance.png'):
        """
        Plot performance metrics for each toxicity task.
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        task_names = results['task_name']
        x_pos = np.arange(len(task_names))
        
        # Plot accuracy
        axes[0].bar(x_pos, results['accuracy'], color='steelblue', alpha=0.8, edgecolor='black')
        axes[0].set_title('Accuracy by Toxicity Assay', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Toxicity Assay', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(task_names, rotation=45, ha='right', fontsize=9)
        axes[0].axhline(y=np.mean(results['accuracy']), color='red', linestyle='--', 
                       linewidth=2, label=f"Average: {np.mean(results['accuracy']):.3f}")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])
        
        # Plot ROC-AUC
        axes[1].bar(x_pos, results['roc_auc'], color='coral', alpha=0.8, edgecolor='black')
        axes[1].set_title('ROC-AUC by Toxicity Assay', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Toxicity Assay', fontsize=12)
        axes[1].set_ylabel('ROC-AUC Score', fontsize=12)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(task_names, rotation=45, ha='right', fontsize=9)
        valid_aucs = [auc for auc in results['roc_auc'] if auc > 0]
        if valid_aucs:
            axes[1].axhline(y=np.mean(valid_aucs), color='red', linestyle='--',
                          linewidth=2, label=f"Average: {np.mean(valid_aucs):.3f}")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Task performance plot saved to: {save_path}")
        plt.show()
        plt.close()
    
    def save_model(self, filepath='toxicology_model.h5'):
        """Save the trained model to disk."""
        self.model.save(filepath)
        print(f"\nModel saved to: {filepath}")
    
    def load_model(self, filepath='toxicology_model.h5'):
        """Load a trained model from disk."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"\nModel loaded from: {filepath}")


def main():
    """
    Main function to run the toxicology prediction pipeline.
    """
    print("\n" + "="*70)
    print(" " * 15 + "TOXICOLOGY NEURAL NETWORK PREDICTOR")
    print(" " * 20 + "Tox21 Dataset Analysis")
    print("="*70)
    
    # Step 1: Load the Tox21 Dataset
    print("\n[STEP 1/9] Loading Tox21 Dataset...")
    print("-" * 70)
    
    try:
        _, (train, valid, test), _ = dc.molnet.load_tox21()
        train_X, train_y, train_w = train.X, train.y, train.w
        valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
        test_X, test_y, test_w = test.X, test.y, test.w
        
        print(f"\nDataset loaded successfully!")
        print(f"  Training set: {train_X.shape[0]:,} samples")
        print(f"  Validation set: {valid_X.shape[0]:,} samples")
        print(f"  Test set: {test_X.shape[0]:,} samples")
        print(f"  Number of features: {train_X.shape[1]:,}")
        print(f"  Number of tasks: {train_y.shape[1]}")
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("Make sure DeepChem is installed: pip install deepchem")
        return
    
    # Step 2: Analyze Dataset
    print("\n[STEP 2/9] Analyzing Dataset Characteristics...")
    print("-" * 70)
    
    print(f"\nDataset Summary:")
    print(f"  Total compounds: {train_X.shape[0] + valid_X.shape[0] + test_X.shape[0]:,}")
    print(f"  Molecular descriptors per compound: {train_X.shape[1]:,}")
    print(f"  Toxicity assays: {train_y.shape[1]}")
    
    # Analyze class distribution
    print(f"\nClass Distribution (Training Set):")
    task_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                  'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                  'SR-HSE', 'SR-MMP', 'SR-p53']
    
    for i, task_name in enumerate(task_names):
        valid_samples = train_w[:, i] != 0
        if valid_samples.sum() > 0:
            toxic_count = (train_y[valid_samples, i] == 1).sum()
            total_count = valid_samples.sum()
            toxic_pct = (toxic_count / total_count) * 100
            print(f"  {task_name:<20} Toxic: {toxic_count:>4}/{total_count:>5} ({toxic_pct:>5.1f}%)")
    
    # Step 3: Build the Neural Network
    print("\n[STEP 3/9] Building Neural Network Architecture...")
    print("-" * 70)
    
    predictor = ToxicologyPredictor(input_dim=train_X.shape[1], num_tasks=train_y.shape[1])
    predictor.build_model(hidden_layers=[1000, 500, 250], dropout_rate=0.3)
    
    # Step 4: Display Model Configuration
    print("\n[STEP 4/9] Reviewing Model Configuration...")
    print("-" * 70)
    
    print(f"\nTraining Hyperparameters:")
    print(f"  Epochs: 50")
    print(f"  Batch Size: 128")
    print(f"  Learning Rate: 0.001")
    print(f"  Optimizer: Adam")
    print(f"  Loss Function: Binary Cross-Entropy")
    print(f"  Regularization: Dropout (0.3), L2 (0.001), Batch Normalization")
    print(f"  Early Stopping: Enabled (patience=10)")
    print(f"  Learning Rate Reduction: Enabled (factor=0.5, patience=5)")
    
    # Step 5: Train the Model
    print("\n[STEP 5/9] Training Neural Network...")
    print("-" * 70)
    
    predictor.train(
        train_X, train_y, train_w,
        valid_X, valid_y, valid_w,
        epochs=50,
        batch_size=128,
        verbose=1
    )
    
    # Step 6: Evaluate on Test Set
    print("\n[STEP 6/9] Evaluating Model Performance...")
    print("-" * 70)
    
    results = predictor.evaluate(test_X, test_y, test_w)
    
    # Step 7: Generate Visualizations
    print("\n[STEP 7/9] Generating Visualizations...")
    print("-" * 70)
    
    predictor.plot_training_history('toxicology_training_history.png')
    predictor.plot_loss_curve('toxicology_loss_curve.png')
    predictor.plot_task_performance(results, 'toxicology_task_performance.png')
    
    # Step 8: Make Sample Predictions
    print("\n[STEP 8/9] Making Sample Predictions...")
    print("-" * 70)
    
    # Select a random compound from test set
    sample_idx = np.random.randint(0, len(test_X))
    sample_features = test_X[sample_idx]
    
    print(f"\nPredicting toxicity for test compound #{sample_idx}...")
    predictions = predictor.predict_compound(sample_features)
    
    # Show actual labels if available
    print("\nActual Labels (if available):")
    print("-" * 70)
    for i, task_name in enumerate(predictor.task_names):
        if test_w[sample_idx, i] != 0:  # Label is available
            actual = "TOXIC" if test_y[sample_idx, i] == 1 else "NON-TOXIC"
            print(f"{task_name:<25} {actual}")
    
    # Step 9: Save Results and Model
    print("\n[STEP 9/9] Saving Results and Model...")
    print("-" * 70)
    
    # Save model
    predictor.save_model('toxicology_neural_network_model.h5')
    
    # Save evaluation results
    results_df = pd.DataFrame(results)
    results_df.to_csv('toxicology_evaluation_results.csv', index=False)
    print(f"Evaluation results saved to: toxicology_evaluation_results.csv")
    
    # Generate summary report
    print("\nCreating summary report...")
    summary = {
        'Model Architecture': f"{len([1000, 500, 250])} hidden layers",
        'Total Parameters': f"{predictor.model.count_params():,}",
        'Training Samples': f"{len(train_X):,}",
        'Test Samples': f"{len(test_X):,}",
        'Average Accuracy': f"{np.mean(results['accuracy']):.4f}",
        'Average ROC-AUC': f"{np.mean([auc for auc in results['roc_auc'] if auc > 0]):.4f}",
        'Number of Tasks': f"{train_y.shape[1]}"
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('toxicology_model_summary.csv', index=False)
    print(f"Model summary saved to: toxicology_model_summary.csv")
    
    print("\n" + "="*70)
    print(" " * 20 + "ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated Files:")
    print("  1. toxicology_neural_network_model.h5 - Trained model")
    print("  2. toxicology_training_history.png - Training metrics visualization")
    print("  3. toxicology_loss_curve.png - Detailed loss curve")
    print("  4. toxicology_task_performance.png - Task-wise performance metrics")
    print("  5. toxicology_evaluation_results.csv - Detailed evaluation results")
    print("  6. toxicology_model_summary.csv - Model summary and performance")
    print("  7. logs/fit/ - TensorBoard logs for model convergence tracking")
    print("\n" + "="*70)
    print("\n📊 To track model convergence with TensorBoard:")
    print("  1. Open a new terminal")
    print("  2. Activate virtual environment: source tox_env/bin/activate")
    print("  3. Run: tensorboard --logdir=logs/fit")
    print("  4. Open browser: http://localhost:6006")
    print("  5. View SCALARS tab for loss/accuracy convergence")
    print("  6. View GRAPHS tab for model architecture")
    print("="*70)


if __name__ == "__main__":
    main()

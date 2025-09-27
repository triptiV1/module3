#!/usr/bin/env python3
"""
TensorFlow CNN Demo Execution Script
CSC525 Assignment - Convolutional Neural Network for MNIST Digit Recognition

This script executes the CNN example and captures results for assignment documentation.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

print("="*60)
print("TensorFlow CNN Demo - MNIST Digit Recognition")
print("="*60)
print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("="*60)

# MNIST dataset parameters
num_classes = 10  # total classes (0-9 digits)

# Training parameters
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 10

# Network parameters
conv1_filters = 32  # number of filters for 1st conv layer
conv2_filters = 64  # number of filters for 2nd conv layer
fc1_units = 1024   # number of neurons for 1st fully-connected layer

print("\nModel Configuration:")
print(f"- Classes: {num_classes}")
print(f"- Learning Rate: {learning_rate}")
print(f"- Training Steps: {training_steps}")
print(f"- Batch Size: {batch_size}")
print(f"- Conv1 Filters: {conv1_filters}")
print(f"- Conv2 Filters: {conv2_filters}")
print(f"- FC1 Units: {fc1_units}")

# Prepare MNIST data
print("\nLoading MNIST Dataset...")
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

# Normalize images value from [0, 255] to [0, 1]
x_train, x_test = x_train / 255., x_test / 255.

print(f"Training samples: {x_train.shape[0]}")
print(f"Testing samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:3]}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Display sample images
print("\nCreating sample visualization...")
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('MNIST Sample Images', fontsize=16)
for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(x_train[i], cmap='gray')
    axes[row, col].set_title(f'Label: {y_train[i]}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
print("✓ Sample images saved as 'mnist_samples.png'")

# Use tf.data API to shuffle and batch data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Create TF Model
class ConvNet(Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolution Layer with 32 filters and a kernel size of 5
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2
        self.maxpool1 = layers.MaxPool2D(2, strides=2)
        
        # Convolution Layer with 64 filters and a kernel size of 3
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2
        self.maxpool2 = layers.MaxPool2D(2, strides=2)
        
        # Flatten the data to a 1-D vector for the fully connected layer
        self.flatten = layers.Flatten()
        
        # Fully connected layer
        self.fc1 = layers.Dense(1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        self.dropout = layers.Dropout(rate=0.5)
        
        # Output layer, class prediction
        self.out = layers.Dense(num_classes)
    
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=is_training)
        x = self.out(x)
        if not is_training:
            # Apply softmax when not training
            x = tf.nn.softmax(x)
        return x

# Build neural network model
print("\nBuilding CNN Model...")
conv_net = ConvNet()

# Cross-Entropy Loss
def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch
    return tf.reduce_mean(loss)

# Accuracy metric
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Adam optimizer
optimizer = tf.optimizers.Adam(learning_rate)

# Optimization process
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation
    with tf.GradientTape() as g:
        # Forward pass
        pred = conv_net(x, is_training=True)
        # Compute loss
        loss = cross_entropy_loss(pred, y)
        
    # Variables to update, i.e. trainable variables
    trainable_variables = conv_net.trainable_variables

    # Compute gradients
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients
    optimizer.apply_gradients(zip(gradients, trainable_variables))

print("✓ Model architecture created")
print("✓ Loss function and optimizer configured")

# Training
print(f"\nStarting Training ({training_steps} steps)...")
print("-" * 50)

training_losses = []
training_accuracies = []
steps = []

# Run training for the given number of steps
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = conv_net(batch_x)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        
        training_losses.append(float(loss))
        training_accuracies.append(float(acc))
        steps.append(step)
        
        print(f"Step: {step:3d}, Loss: {loss:.6f}, Accuracy: {acc:.4f}")

print("-" * 50)
print("✓ Training completed!")

# Test the model
print("\nEvaluating on Test Set...")
test_pred = conv_net(x_test)
test_accuracy = accuracy(test_pred, y_test)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Create training progress visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(steps, training_losses, 'b-', linewidth=2)
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(steps, training_accuracies, 'g-', linewidth=2)
plt.title('Training Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(steps, [acc*100 for acc in training_accuracies], 'r-', linewidth=2)
plt.title('Training Accuracy (%)')
plt.xlabel('Step')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
print("✓ Training progress saved as 'training_progress.png'")

# Test predictions on sample images
print("\nGenerating Prediction Examples...")
sample_indices = [0, 1, 2, 3, 4, 100, 200, 300, 400, 500]
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]
sample_predictions = conv_net(sample_images)
predicted_classes = tf.argmax(sample_predictions, axis=1)

# Create prediction visualization
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
fig.suptitle('CNN Predictions on Test Images', fontsize=16)

for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(sample_images[i], cmap='gray')
    
    true_label = sample_labels[i]
    pred_label = predicted_classes[i].numpy()
    confidence = tf.nn.softmax(sample_predictions[i])[pred_label].numpy()
    
    color = 'green' if true_label == pred_label else 'red'
    axes[row, col].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}', 
                            color=color, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('prediction_examples.png', dpi=150, bbox_inches='tight')
print("✓ Prediction examples saved as 'prediction_examples.png'")

# Model summary
print("\nModel Summary:")
print("=" * 60)
print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Total Parameters: {conv_net.count_params():,}")
print(f"Training Steps: {training_steps}")
print(f"Final Training Loss: {training_losses[-1]:.6f}")
print(f"Final Training Accuracy: {training_accuracies[-1]:.4f}")

# Save results summary
results_summary = f"""
TensorFlow CNN Demo Results
==========================
Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
TensorFlow Version: {tf.__version__}

Dataset: MNIST Handwritten Digits
- Training samples: {x_train.shape[0]:,}
- Testing samples: {x_test.shape[0]:,}
- Image size: {x_train.shape[1]}x{x_train.shape[2]}
- Classes: {num_classes}

Model Architecture:
- Conv2D Layer 1: {conv1_filters} filters, 5x5 kernel, ReLU
- MaxPool2D Layer 1: 2x2 pool, stride 2
- Conv2D Layer 2: {conv2_filters} filters, 3x3 kernel, ReLU
- MaxPool2D Layer 2: 2x2 pool, stride 2
- Dense Layer: {fc1_units} units
- Dropout: 0.5 rate
- Output Layer: {num_classes} units, softmax

Training Configuration:
- Optimizer: Adam (lr={learning_rate})
- Loss: Sparse Categorical Crossentropy
- Batch Size: {batch_size}
- Training Steps: {training_steps}

Results:
- Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
- Total Parameters: {conv_net.count_params():,}
- Final Training Loss: {training_losses[-1]:.6f}
- Final Training Accuracy: {training_accuracies[-1]:.4f}

Generated Files:
- mnist_samples.png: Sample MNIST images
- training_progress.png: Training loss and accuracy curves
- prediction_examples.png: Model predictions on test images
- results_summary.txt: This summary file
"""

with open('results_summary.txt', 'w') as f:
    f.write(results_summary)

print("✓ Results summary saved as 'results_summary.txt'")
print("\n" + "="*60)
print("CNN Demo Execution Complete!")
print("Files generated for assignment documentation:")
print("- mnist_samples.png")
print("- training_progress.png") 
print("- prediction_examples.png")
print("- results_summary.txt")
print("="*60)

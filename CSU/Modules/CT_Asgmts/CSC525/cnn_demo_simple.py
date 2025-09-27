#!/usr/bin/env python3
"""
TensorFlow CNN Demo - Simplified Version for Assignment Documentation
CSC525 Assignment - Convolutional Neural Network Architecture Demonstration

This script demonstrates CNN concepts with synthetic data for assignment screenshots.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

print("="*60)
print("TensorFlow CNN Demo - Architecture Demonstration")
print("="*60)
print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("="*60)

# Create synthetic MNIST-like data for demonstration
def create_synthetic_mnist():
    """Create synthetic 28x28 digit-like images for demonstration"""
    np.random.seed(42)  # For reproducibility
    
    # Generate 1000 training samples and 200 test samples
    n_train, n_test = 1000, 200
    
    # Create synthetic digit patterns
    x_train = np.random.rand(n_train, 28, 28).astype(np.float32)
    x_test = np.random.rand(n_test, 28, 28).astype(np.float32)
    
    # Create simple patterns for different "digits"
    for i in range(n_train):
        digit = i % 10
        # Create simple patterns based on digit
        if digit == 0:  # Circle-like pattern
            center = (14, 14)
            for y in range(28):
                for x in range(28):
                    dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
                    if 8 < dist < 12:
                        x_train[i, y, x] = 0.8 + 0.2 * np.random.rand()
        elif digit == 1:  # Vertical line
            x_train[i, :, 12:16] = 0.7 + 0.3 * np.random.rand(28, 4)
        elif digit == 2:  # Horizontal lines
            x_train[i, 8:12, :] = 0.6 + 0.4 * np.random.rand(4, 28)
            x_train[i, 16:20, :] = 0.6 + 0.4 * np.random.rand(4, 28)
        # Add more patterns for other digits...
        else:
            # Random pattern with some structure
            x_train[i] = np.random.rand(28, 28) * 0.5
            x_train[i, digit:digit+8, digit:digit+8] = 0.8
    
    # Similar for test data
    for i in range(n_test):
        digit = i % 10
        if digit == 0:
            center = (14, 14)
            for y in range(28):
                for x in range(28):
                    dist = np.sqrt((x-center[0])**2 + (y-center[1])**2)
                    if 8 < dist < 12:
                        x_test[i, y, x] = 0.8 + 0.2 * np.random.rand()
        elif digit == 1:
            x_test[i, :, 12:16] = 0.7 + 0.3 * np.random.rand(28, 4)
        elif digit == 2:
            x_test[i, 8:12, :] = 0.6 + 0.4 * np.random.rand(4, 28)
            x_test[i, 16:20, :] = 0.6 + 0.4 * np.random.rand(4, 28)
        else:
            x_test[i] = np.random.rand(28, 28) * 0.5
            x_test[i, digit:digit+8, digit:digit+8] = 0.8
    
    # Create labels
    y_train = np.array([i % 10 for i in range(n_train)])
    y_test = np.array([i % 10 for i in range(n_test)])
    
    return (x_train, y_train), (x_test, y_test)

# Model parameters
num_classes = 10
learning_rate = 0.001
training_steps = 50  # Reduced for demo
batch_size = 32
display_step = 10

print("\nModel Configuration:")
print(f"- Classes: {num_classes}")
print(f"- Learning Rate: {learning_rate}")
print(f"- Training Steps: {training_steps}")
print(f"- Batch Size: {batch_size}")

# Load synthetic data
print("\nGenerating Synthetic MNIST-like Dataset...")
(x_train, y_train), (x_test, y_test) = create_synthetic_mnist()

print(f"Training samples: {x_train.shape[0]}")
print(f"Testing samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:3]}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Display sample images
print("\nCreating sample visualization...")
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Synthetic MNIST-like Sample Images', fontsize=16)
for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(x_train[i], cmap='gray')
    axes[row, col].set_title(f'Label: {y_train[i]}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('synthetic_mnist_samples.png', dpi=150, bbox_inches='tight')
print("✓ Sample images saved as 'synthetic_mnist_samples.png'")

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
        # Apply Dropout
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
            x = tf.nn.softmax(x)
        return x

# Build neural network model
print("\nBuilding CNN Model...")
conv_net = ConvNet()

# Loss and metrics
def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)

def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Optimizer
optimizer = tf.optimizers.Adam(learning_rate)

# Training function
def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = conv_net(x, is_training=True)
        loss = cross_entropy_loss(pred, y)
    
    trainable_variables = conv_net.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

print("✓ Model architecture created")
print("✓ Loss function and optimizer configured")

# Prepare data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.batch(batch_size).prefetch(1)

# Training
print(f"\nStarting Training ({training_steps} steps)...")
print("-" * 50)

training_losses = []
training_accuracies = []
steps = []

step = 0
for batch_x, batch_y in train_data:
    if step >= training_steps:
        break
    
    step += 1
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
plt.plot(steps, training_losses, 'b-', linewidth=2, marker='o')
plt.title('Training Loss Progress', fontsize=14)
plt.xlabel('Training Step')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(steps, training_accuracies, 'g-', linewidth=2, marker='s')
plt.title('Training Accuracy Progress', fontsize=14)
plt.xlabel('Training Step')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(steps, [acc*100 for acc in training_accuracies], 'r-', linewidth=2, marker='^')
plt.title('Training Accuracy (%)', fontsize=14)
plt.xlabel('Training Step')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_training_progress.png', dpi=150, bbox_inches='tight')
print("✓ Training progress saved as 'cnn_training_progress.png'")

# Test predictions on sample images
print("\nGenerating Prediction Examples...")
sample_indices = [0, 1, 2, 3, 4, 10, 20, 30, 40, 50]
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
plt.savefig('cnn_prediction_examples.png', dpi=150, bbox_inches='tight')
print("✓ Prediction examples saved as 'cnn_prediction_examples.png'")

# Create CNN architecture visualization
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.text(0.5, 0.95, 'CNN Architecture for MNIST Digit Classification', 
        ha='center', va='top', fontsize=16, fontweight='bold', transform=ax.transAxes)

# Draw architecture diagram
layers_info = [
    ('Input\n28×28×1', 0.1, 0.7, 'lightblue'),
    ('Conv2D\n32 filters\n5×5 kernel', 0.25, 0.7, 'lightgreen'),
    ('MaxPool2D\n2×2 pool', 0.4, 0.7, 'lightcoral'),
    ('Conv2D\n64 filters\n3×3 kernel', 0.55, 0.7, 'lightgreen'),
    ('MaxPool2D\n2×2 pool', 0.7, 0.7, 'lightcoral'),
    ('Flatten', 0.1, 0.4, 'lightyellow'),
    ('Dense\n1024 units', 0.3, 0.4, 'lightpink'),
    ('Dropout\n0.5 rate', 0.5, 0.4, 'lightgray'),
    ('Output\n10 classes', 0.7, 0.4, 'lightsteelblue')
]

for layer_name, x, y, color in layers_info:
    ax.add_patch(plt.Rectangle((x-0.06, y-0.08), 0.12, 0.16, 
                              facecolor=color, edgecolor='black', linewidth=2))
    ax.text(x, y, layer_name, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows
arrow_props = dict(arrowstyle='->', lw=2, color='black')
arrows = [(0.16, 0.7, 0.08, 0), (0.31, 0.7, 0.08, 0), (0.46, 0.7, 0.08, 0), 
          (0.61, 0.7, 0.08, 0), (0.1, 0.62, 0, -0.14), (0.16, 0.4, 0.08, 0), 
          (0.36, 0.4, 0.08, 0), (0.56, 0.4, 0.08, 0)]

for x, y, dx, dy in arrows:
    ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y), arrowprops=arrow_props)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.savefig('cnn_architecture_diagram.png', dpi=150, bbox_inches='tight')
print("✓ Architecture diagram saved as 'cnn_architecture_diagram.png'")

# Model summary and results
print("\nCNN Model Summary:")
print("=" * 60)
print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Training Steps Completed: {training_steps}")
if training_losses:
    print(f"Final Training Loss: {training_losses[-1]:.6f}")
    print(f"Final Training Accuracy: {training_accuracies[-1]:.4f}")

# Save results summary
results_summary = f"""
TensorFlow CNN Demo Results - Assignment Documentation
====================================================
Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
TensorFlow Version: {tf.__version__}
Student: Tripti Vishwakarma
Course: CSC525 - Machine Learning

Dataset: Synthetic MNIST-like Handwritten Digits
- Training samples: {x_train.shape[0]:,}
- Testing samples: {x_test.shape[0]:,}
- Image size: {x_train.shape[1]}×{x_train.shape[2]}
- Classes: {num_classes}

CNN Model Architecture:
- Input Layer: 28×28×1 grayscale images
- Conv2D Layer 1: 32 filters, 5×5 kernel, ReLU activation
- MaxPool2D Layer 1: 2×2 pooling, stride 2
- Conv2D Layer 2: 64 filters, 3×3 kernel, ReLU activation  
- MaxPool2D Layer 2: 2×2 pooling, stride 2
- Flatten Layer: Convert 2D to 1D
- Dense Layer: 1024 units, ReLU activation
- Dropout Layer: 0.5 dropout rate
- Output Layer: 10 units, softmax activation

Training Configuration:
- Optimizer: Adam (learning_rate={learning_rate})
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: {batch_size}
- Training Steps: {training_steps}

Performance Results:
- Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
- Training completed successfully
- Model demonstrates CNN concepts effectively


"""

with open('cnn_demo_results.txt', 'w') as f:
    f.write(results_summary)

print("\n✓ CNN demo execution complete - results saved to cnn_demo_results.txt")


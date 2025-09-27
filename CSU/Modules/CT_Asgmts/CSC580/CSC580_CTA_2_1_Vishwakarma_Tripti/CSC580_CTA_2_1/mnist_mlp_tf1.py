#!/usr/bin/env python3
"""
MNIST MLP (TF1-style graph) with configurable hyperparameters and utilities.
- Uses tf.compat.v1 to match placeholder/session workflow from assignment prompt
- Supports 1 or 2 hidden layers
- Saves misclassified examples to outputs/misclassified/
- Can be imported (run_experiment) or executed as a CLI
"""
import os
import argparse
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Use TF1 style
tf.compat.v1.disable_eager_execution()


def load_mnist(normalize=True, one_hot=True, seed=42):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_images = x_train.reshape(60000, 784).astype('float32')
    test_images = x_test.reshape(10000, 784).astype('float32')

    if normalize:
        train_images /= 255.0
        test_images /= 255.0

    if one_hot:
        y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
        y_test_oh = tf.keras.utils.to_categorical(y_test, 10)
    else:
        y_train_oh, y_test_oh = y_train, y_test

    return (train_images, y_train_oh), (test_images, y_test_oh), (x_test, y_test)


def build_graph(hidden_units=512, learning_rate=0.5, layers=1, seed=42):
    rng = np.random.RandomState(seed)
    graph = tf.Graph()
    with graph.as_default():
        tf.compat.v1.set_random_seed(seed)

        input_images = tf.compat.v1.placeholder(tf.float32, shape=[None, 784], name='input_images')
        target_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, 10], name='target_labels')

        # Weights and biases
        input_weights = tf.Variable(tf.random.truncated_normal([784, hidden_units], stddev=0.1, seed=seed), name='input_weights')
        input_biases = tf.Variable(tf.zeros([hidden_units]), name='input_biases')

        hidden = tf.nn.relu(tf.matmul(input_images, input_weights) + input_biases, name='hidden_layer_1')

        if layers == 2:
            hidden2_weights = tf.Variable(tf.random.truncated_normal([hidden_units, hidden_units], stddev=0.1, seed=seed+1), name='hidden2_weights')
            hidden2_biases = tf.Variable(tf.zeros([hidden_units]), name='hidden2_biases')
            hidden = tf.nn.relu(tf.matmul(hidden, hidden2_weights) + hidden2_biases, name='hidden_layer_2')

        hidden_weights = tf.Variable(tf.random.truncated_normal([hidden_units, 10], stddev=0.1, seed=seed+2), name='hidden_weights')
        hidden_biases = tf.Variable(tf.zeros([10]), name='hidden_biases')

        logits = tf.matmul(hidden, hidden_weights) + hidden_biases

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target_labels), name='loss')
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(target_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        saver = tf.compat.v1.train.Saver(max_to_keep=1)

    return graph, {'input_images': input_images, 'target_labels': target_labels, 'logits': logits, 'loss': loss, 'optimizer': optimizer, 'accuracy': accuracy, 'saver': saver}


def save_misclassified_images(xs_flat, y_true_digits, y_pred_digits, limit=20, out_dir='outputs/misclassified'):
    os.makedirs(out_dir, exist_ok=True)
    ms_idx = np.where(y_true_digits != y_pred_digits)[0]
    saved = []
    for i, idx in enumerate(ms_idx[:limit]):
        img = xs_flat[idx].reshape(28, 28)
        true_d = int(y_true_digits[idx])
        pred_d = int(y_pred_digits[idx])
        fname = os.path.join(out_dir, f'misclassified_{i:03d}_true{true_d}_pred{pred_d}.png')
        plt.imsave(fname, img, cmap='gray_r')
        saved.append(fname)
    return saved


def run_experiment(hidden_units=512, learning_rate=0.5, batch_size=100, epochs=20, layers=1, seed=42, save_miscl=20, work_dir='outputs'):
    os.makedirs(work_dir, exist_ok=True)
    (x_train, y_train), (x_test, y_test), (x_test_img, y_test_digits) = load_mnist()

    graph, nodes = build_graph(hidden_units=hidden_units, learning_rate=learning_rate, layers=layers, seed=seed)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        train_size = x_train.shape[0]
        period = train_size // batch_size

        for e in range(epochs):
            idxs = np.random.permutation(train_size)
            X_random = x_train[idxs]
            Y_random = y_train[idxs]
            for i in range(period):
                batch_X = X_random[i * batch_size:(i+1) * batch_size]
                batch_Y = Y_random[i * batch_size:(i+1) * batch_size]
                sess.run(nodes['optimizer'], feed_dict={nodes['input_images']: batch_X, nodes['target_labels']: batch_Y})

            acc = sess.run(nodes['accuracy'], feed_dict={nodes['input_images']: x_test, nodes['target_labels']: y_test})
            print(f"Epoch {e+1}/{epochs} - Test Accuracy: {acc:.4f}")

        # Final evaluation and misclassified
        logits = sess.run(nodes['logits'], feed_dict={nodes['input_images']: x_test})
        y_pred_digits = np.argmax(logits, axis=1)
        test_acc = np.mean(y_pred_digits == y_test_digits)
        print(f"Final Test Accuracy: {test_acc:.4f}")

        mis_paths = save_misclassified_images(x_test, y_test_digits, y_pred_digits, limit=save_miscl, out_dir=os.path.join(work_dir, 'misclassified')) if save_miscl else []

    result = {
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'layers': layers,
        'seed': seed,
        'test_accuracy': float(test_acc),
        'misclassified_saved': mis_paths,
        'timestamp': datetime.now().isoformat(timespec='seconds')
    }
    # Save a JSON log
    with open(os.path.join(work_dir, 'last_run.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description='MNIST MLP (TF1-style via tf.compat.v1)')
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--layers', type=int, choices=[1, 2], default=1, help='Number of hidden layers (1 or 2)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_miscl', type=int, default=20, help='How many misclassified examples to save (0 to disable)')
    parser.add_argument('--work_dir', type=str, default='outputs')
    args = parser.parse_args()

    res = run_experiment(hidden_units=args.hidden_units,
                         learning_rate=args.learning_rate,
                         batch_size=args.batch_size,
                         epochs=args.epochs,
                         layers=args.layers,
                         seed=args.seed,
                         save_miscl=args.save_miscl,
                         work_dir=args.work_dir)
    print('\nSummary:')
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()

import numpy as np

# Activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Normalize data function
def normalize_data(data, scale_factor=None):
    if scale_factor is None:
        scale_factor = np.max(data)
    return data / scale_factor, scale_factor

# Function to train and predict
def train_and_predict(sequence):
    # Initialize parameters
    input_size = 1
    hidden_size1 = 30  # Increased number of neurons
    hidden_size2 = 20  # Added second hidden layer
    output_size = 1
    learning_rate = 0.01
    epochs = 2000

    # Initialize weights and biases
    weights_input_hidden1 = np.random.rand(input_size, hidden_size1) * 0.1
    weights_hidden1_hidden2 = np.random.rand(hidden_size1, hidden_size2) * 0.1
    weights_hidden2_output = np.random.rand(hidden_size2, output_size) * 0.1
    bias_hidden1 = np.random.rand(1, hidden_size1) * 0.1
    bias_hidden2 = np.random.rand(1, hidden_size2) * 0.1
    bias_output = np.random.rand(1, output_size) * 0.1

    # Prepare training data
    X_train = np.array(sequence[:-1]).reshape(-1, 1)
    y_train = np.array(sequence[1:]).reshape(-1, 1)

    # Normalize data
    X_train_norm, scale_factor = normalize_data(X_train)
    y_train_norm, _ = normalize_data(y_train, scale_factor)

    # Training loop
    for epoch in range(epochs):
        # Feedforward
        hidden_input1 = np.dot(X_train_norm, weights_input_hidden1) + bias_hidden1
        hidden_output1 = relu(hidden_input1)

        hidden_input2 = np.dot(hidden_output1, weights_hidden1_hidden2) + bias_hidden2
        hidden_output2 = relu(hidden_input2)

        final_input = np.dot(hidden_output2, weights_hidden2_output) + bias_output
        final_output = relu(final_input)

        # Calculate error
        error = y_train_norm - final_output
        loss = np.mean(error**2)

        # Backpropagation
        d_output = error * relu_derivative(final_output)
        d_hidden2 = np.dot(d_output, weights_hidden2_output.T) * relu_derivative(hidden_output2)
        d_hidden1 = np.dot(d_hidden2, weights_hidden1_hidden2.T) * relu_derivative(hidden_output1)

        weights_hidden2_output += np.dot(hidden_output2.T, d_output) * learning_rate
        weights_hidden1_hidden2 += np.dot(hidden_output1.T, d_hidden2) * learning_rate
        weights_input_hidden1 += np.dot(X_train_norm.T, d_hidden1) * learning_rate
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        bias_hidden2 += np.sum(d_hidden2, axis=0, keepdims=True) * learning_rate
        bias_hidden1 += np.sum(d_hidden1, axis=0, keepdims=True) * learning_rate

        # Print the loss every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Predict the next number in the sequence
    user_input = np.array([sequence[-1]]).reshape(-1, 1)
    user_input_norm, _ = normalize_data(user_input, scale_factor)

    hidden_test_input1 = np.dot(user_input_norm, weights_input_hidden1) + bias_hidden1
    hidden_test_output1 = relu(hidden_test_input1)

    hidden_test_input2 = np.dot(hidden_test_output1, weights_hidden1_hidden2) + bias_hidden2
    hidden_test_output2 = relu(hidden_test_input2)

    final_test_input = np.dot(hidden_test_output2, weights_hidden2_output) + bias_output
    final_test_output = relu(final_test_input)

    # Denormalize the output
    predicted_output = final_test_output * scale_factor
    
    print(f'Predicted next number in the sequence is: {predicted_output[0][0]}')

# Get user input for sequence
try:
    sequence_input = input("Enter a sequence of numbers separated by spaces: ")
    sequence = [float(num) for num in sequence_input.split()]
    if len(sequence) < 2:
        print("Please enter at least two numbers in the sequence.")
    else:
        train_and_predict(sequence)
except ValueError:
    print("Invalid input. Please enter numerical values separated by spaces.")

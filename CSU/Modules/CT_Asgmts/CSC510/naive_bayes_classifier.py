import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ===== Step 1: Create a Dataset =====
# Example dataset: binary features with binary class labels
data = pd.DataFrame({
    "Feature1": [0, 0, 1, 1, 0, 1, 1, 0],
    "Feature2": [0, 1, 0, 1, 0, 0, 1, 1],
    "Label": [0, 1, 1, 0, 0, 1, 1, 0]
})

# Features (X) and labels (y)
X = data[["Feature1", "Feature2"]]
y = data["Label"]

# ===== Step 2: Split Dataset =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== Step 3: Create Frequency Table =====
def create_frequency_table(X, y):
    freq_table = {}
    for label in np.unique(y):
        freq_table[label] = X[y == label].apply(pd.Series.value_counts).fillna(0).astype(int)
    return freq_table

freq_table = create_frequency_table(X_train, y_train)
print("=== Frequency Table ===")
for label, table in freq_table.items():
    print(f"Class {label}:\n{table}")

# ===== Step 4: Create Likelihood Table =====
def create_likelihood_table(freq_table, laplace_smoothing=1):
    likelihood_table = {}
    for label, freq in freq_table.items():
        likelihood_table[label] = (freq + laplace_smoothing) / (freq.sum() + laplace_smoothing * len(freq.columns))
        likelihood_table[label].index.name = "FeatureValue"  # Name the index for clarity
    return likelihood_table

likelihood_table = create_likelihood_table(freq_table)
print("\n=== Likelihood Table ===")
for label, table in likelihood_table.items():
    print(f"Class {label}:\n{table}")

# ===== Step 5: Calculate Posterior Probabilities =====
def calculate_posterior(likelihood_table, priors, input_features):
    posteriors = {}
    for label, likelihood in likelihood_table.items():
        posterior = priors[label]
        for feature_index, value in enumerate(input_features):
            feature_name = f"Feature{feature_index + 1}"  # Match column names in likelihood table
            if feature_name in likelihood.columns:
                if value in likelihood.index:
                    posterior *= likelihood.at[value, feature_name]
                else:
                    raise ValueError(f"Value {value} not found in likelihood table for feature '{feature_name}' in class {label}.")
            else:
                raise ValueError(f"Feature '{feature_name}' not found in likelihood table for class {label}.")
        posteriors[label] = posterior
    return posteriors

# Calculate class priors
priors = {label: len(y_train[y_train == label]) / len(y_train) for label in np.unique(y_train)}

# Example input features for prediction
input_features = [1, 0]  # Feature1=1, Feature2=0
print("\n=== Priors ===")
print(priors)

print("\n=== Input Features ===")
print(input_features)

posterior_probs = calculate_posterior(likelihood_table, priors, input_features)
print("\n=== Posterior Probabilities ===")
print(posterior_probs)

# ===== Step 6: Make Prediction =====
predicted_class = max(posterior_probs, key=posterior_probs.get)
print("\nPredicted Class:", predicted_class)

# ===== Step 7: Custom Input Prediction =====
print("\n=== Custom Input Prediction ===")
def get_custom_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value in [0, 1]:
                return value
            else:
                print("Invalid input. Please enter 0 or 1.")
        except ValueError:
            print("Invalid input. Please enter 0 or 1.")

custom_feature1 = get_custom_input("Enter value for Feature1 (0 or 1): ")
custom_feature2 = get_custom_input("Enter value for Feature2 (0 or 1): ")

# Predict for custom input
custom_input_features = [custom_feature1, custom_feature2]
custom_posterior_probs = calculate_posterior(likelihood_table, priors, custom_input_features)
custom_predicted_class = max(custom_posterior_probs, key=custom_posterior_probs.get)

print("Predicted Class:", custom_predicted_class)
print("Class Probabilities:", custom_posterior_probs)

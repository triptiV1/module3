#!/usr/bin/env python3
"""
KNN Classifier for Iris Dataset
CSC525 Assignment - Option #1

This script implements a K-Nearest Neighbors classifier from scratch to predict
iris species based on sepal and petal measurements.

Usage:
    python knn_iris_classifier.py <sepal_length> <sepal_width> <petal_length> <petal_width>

Example:
    python knn_iris_classifier.py 5.1 3.5 1.4 0.2
"""

import sys
import csv
import math
from collections import Counter


class KNNClassifier:
    """K-Nearest Neighbors Classifier implementation"""
    
    def __init__(self, k=3):
        """
        Initialize KNN classifier
        
        Args:
            k (int): Number of nearest neighbors to consider
        """
        self.k = k
        self.training_data = []
        self.training_labels = []
    
    def load_data(self, filename):
        """
        Load training data from CSV file
        
        Args:
            filename (str): Path to the CSV file containing iris data
        """
        try:
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)  # Skip header row
                
                for row in csv_reader:
                    # Extract features (first 4 columns) and convert to float
                    features = [float(row[i]) for i in range(4)]
                    # Extract label (last column)
                    label = row[4]
                    
                    self.training_data.append(features)
                    self.training_labels.append(label)
                    
            print(f"Loaded {len(self.training_data)} training samples")
            
        except FileNotFoundError:
            print(f"Error: Could not find file '{filename}'")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def euclidean_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points
        
        Args:
            point1 (list): First point coordinates
            point2 (list): Second point coordinates
            
        Returns:
            float: Euclidean distance between the points
        """
        if len(point1) != len(point2):
            raise ValueError("Points must have the same number of dimensions")
        
        distance = 0
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
        
        return math.sqrt(distance)
    
    def predict(self, query_point):
        """
        Predict the class of a query point using KNN algorithm
        
        Args:
            query_point (list): Features of the point to classify
            
        Returns:
            str: Predicted class label
        """
        if not self.training_data:
            raise ValueError("No training data loaded. Call load_data() first.")
        
        # Step 1: Calculate distances to all training points
        distances = []
        for i in range(len(self.training_data)):
            distance = self.euclidean_distance(query_point, self.training_data[i])
            distances.append((distance, self.training_labels[i]))
        
        # Step 2: Sort distances in ascending order
        distances.sort(key=lambda x: x[0])
        
        # Step 3: Get top k nearest neighbors
        k_nearest = distances[:self.k]
        
        # Step 4: Extract labels of k nearest neighbors
        k_labels = [label for _, label in k_nearest]
        
        # Step 5: Vote for most frequent class
        label_counts = Counter(k_labels)
        predicted_class = label_counts.most_common(1)[0][0]
        
        # Print debugging information
        print(f"\nKNN Classification Results (k={self.k}):")
        print(f"Query point: {query_point}")
        print(f"Top {self.k} nearest neighbors:")
        for i, (dist, label) in enumerate(k_nearest):
            print(f"  {i+1}. Distance: {dist:.4f}, Class: {label}")
        print(f"Class votes: {dict(label_counts)}")
        
        return predicted_class
    
    def get_accuracy(self, test_data, test_labels):
        """
        Calculate accuracy on test data (for validation purposes)
        
        Args:
            test_data (list): Test feature vectors
            test_labels (list): True labels for test data
            
        Returns:
            float: Accuracy as a percentage
        """
        correct = 0
        total = len(test_data)
        
        for i in range(total):
            predicted = self.predict(test_data[i])
            if predicted == test_labels[i]:
                correct += 1
        
        return (correct / total) * 100


def validate_input(args):
    """
    Validate command line arguments
    
    Args:
        args (list): Command line arguments
        
    Returns:
        list: Validated feature values as floats
    """
    if len(args) != 4:
        print("Error: Please provide exactly 4 floating point numbers")
        print("Usage: python knn_iris_classifier.py <sepal_length> <sepal_width> <petal_length> <petal_width>")
        print("Example: python knn_iris_classifier.py 5.1 3.5 1.4 0.2")
        sys.exit(1)
    
    try:
        features = [float(arg) for arg in args]
        
        # Basic validation - check for reasonable ranges
        if any(f < 0 for f in features):
            print("Warning: Negative values detected. Please check your input.")
        
        if features[0] > 10 or features[1] > 10:  # Sepal measurements
            print("Warning: Unusually large sepal measurements detected.")
        
        if features[2] > 10 or features[3] > 10:  # Petal measurements
            print("Warning: Unusually large petal measurements detected.")
        
        return features
        
    except ValueError:
        print("Error: All arguments must be valid floating point numbers")
        sys.exit(1)


def get_user_input():
    """Get iris measurements from user input"""
    print("Please enter the iris measurements:")
    
    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))
        
        features = [sepal_length, sepal_width, petal_length, petal_width]
        
        # Basic validation - check for reasonable ranges
        if any(f < 0 for f in features):
            print("Warning: Negative values detected. Please check your input.")
        
        if features[0] > 10 or features[1] > 10:  # Sepal measurements
            print("Warning: Unusually large sepal measurements detected.")
        
        if features[2] > 10 or features[3] > 10:  # Petal measurements
            print("Warning: Unusually large petal measurements detected.")
        
        return features
        
    except ValueError:
        print("Error: Please enter valid floating point numbers")
        return None
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)


def main():
    """Main function to run the KNN classifier"""
    
    print("=" * 60)
    print("KNN IRIS CLASSIFIER")
    print("=" * 60)
    
    # Check if command line arguments are provided (for backward compatibility)
    if len(sys.argv) == 5:
        # Use command line arguments
        query_features = validate_input(sys.argv[1:])
    else:
        # Get input interactively
        if len(sys.argv) > 1:
            print("Note: Expected 4 arguments for command line usage.")
            print("Switching to interactive input mode.\n")
        
        query_features = get_user_input()
        if query_features is None:
            sys.exit(1)
    
    # Initialize classifier with k=3 (commonly used default)
    classifier = KNNClassifier(k=3)
    
    # Load training data
    print("Loading training data from iris.csv...")
    classifier.load_data('iris.csv')
    
    # Make prediction
    print(f"\nClassifying iris with features:")
    print(f"  Sepal Length: {query_features[0]} cm")
    print(f"  Sepal Width:  {query_features[1]} cm")
    print(f"  Petal Length: {query_features[2]} cm")
    print(f"  Petal Width:  {query_features[3]} cm")
    
    predicted_class = classifier.predict(query_features)
    
    print(f"\n{'='*60}")
    print(f"PREDICTION: {predicted_class}")
    print(f"{'='*60}")
    
    # Provide some context about the prediction
    species_info = {
        'Iris-setosa': 'Setosa iris - typically has smaller petals and is easily distinguishable',
        'Iris-versicolor': 'Versicolor iris - medium-sized flowers with moderate petal dimensions',
        'Iris-virginica': 'Virginica iris - typically has the largest petals and sepals'
    }
    
    if predicted_class in species_info:
        print(f"\nAbout {predicted_class}:")
        print(f"  {species_info[predicted_class]}")


if __name__ == "__main__":
    main()

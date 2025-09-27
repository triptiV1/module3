#!/usr/bin/env python3
"""
Polynomial Regression Salary Predictor
=====================================

This program implements polynomial regression to predict employee salaries
based on years of experience using the Salary_Data.csv dataset.

Author: Tripti Vishwakarma
Course: CSC525 - Machine Learning
Assignment: Polynomial Regression Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class PolynomialRegressionPredictor:
    """
    A class to implement polynomial regression for salary prediction.
    """
    
    def __init__(self, degree=2):
        """
        Initialize the polynomial regression predictor.
        
        Args:
            degree (int): Degree of polynomial features (default: 2)
        """
        self.degree = degree
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = None
        
    def load_data(self, filepath):
        """
        Load salary data from CSV file.
        
        Args:
            filepath (str): Path to the CSV file
        """
        try:
            self.data = pd.read_csv(filepath)
            print("✓ Data loaded successfully!")
            print(f"Dataset shape: {self.data.shape}")
            print("\nFirst 5 rows:")
            print(self.data.head())
            print(f"\nDataset statistics:")
            print(self.data.describe())
            return True
        except FileNotFoundError:
            print(f"Error: File {filepath} not found!")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for training and testing.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        if self.data is None:
            print("Error: No data loaded. Please load data first.")
            return False
        
        # Extract features and target
        X = self.data[['YearsExperience']].values
        y = self.data['Salary'].values
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✓ Data prepared successfully!")
        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")
        return True
    
    def train_model(self):
        """
        Train the polynomial regression model.
        """
        if self.X_train is None:
            print("Error: No training data available. Please prepare data first.")
            return False
        
        # Create polynomial regression pipeline
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree)),
            ('linear', LinearRegression())
        ])
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        print(f"✓ Polynomial regression model (degree {self.degree}) trained successfully!")
        return True
    
    def evaluate_model(self):
        """
        Evaluate the model performance on training and testing data.
        """
        if self.model is None:
            print("Error: No trained model available. Please train model first.")
            return
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Training Set Performance:")
        print(f"  Mean Squared Error: ${train_mse:,.2f}")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"\nTesting Set Performance:")
        print(f"  Mean Squared Error: ${test_mse:,.2f}")
        print(f"  R² Score: {test_r2:.4f}")
        
        # Get polynomial coefficients
        poly_features = self.model.named_steps['poly']
        linear_model = self.model.named_steps['linear']
        
        print(f"\nPolynomial Coefficients (degree {self.degree}):")
        feature_names = poly_features.get_feature_names_out(['YearsExperience'])
        for i, (coef, name) in enumerate(zip(linear_model.coef_, feature_names)):
            print(f"  {name}: {coef:.2f}")
        print(f"  Intercept: {linear_model.intercept_:.2f}")
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def predict_salary(self, years_experience):
        """
        Predict salary for given years of experience.
        
        Args:
            years_experience (float): Years of experience
            
        Returns:
            float: Predicted salary
        """
        if self.model is None:
            print("Error: No trained model available. Please train model first.")
            return None
        
        # Make prediction
        prediction = self.model.predict([[years_experience]])
        return prediction[0]
    
    def visualize_results(self, save_plot=True):
        """
        Create visualization of the polynomial regression results.
        
        Args:
            save_plot (bool): Whether to save the plot as an image file
        """
        if self.model is None:
            print("Error: No trained model available. Please train model first.")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Training data and polynomial fit
        X_plot = np.linspace(self.data['YearsExperience'].min(), 
                           self.data['YearsExperience'].max(), 100).reshape(-1, 1)
        y_plot = self.model.predict(X_plot)
        
        ax1.scatter(self.X_train, self.y_train, color='blue', alpha=0.6, label='Training Data')
        ax1.scatter(self.X_test, self.y_test, color='red', alpha=0.6, label='Testing Data')
        ax1.plot(X_plot, y_plot, color='green', linewidth=2, 
                label=f'Polynomial Regression (degree {self.degree})')
        ax1.set_xlabel('Years of Experience')
        ax1.set_ylabel('Salary ($)')
        ax1.set_title('Polynomial Regression: Salary vs Experience')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Predictions vs Actual
        y_pred_all = self.model.predict(self.data[['YearsExperience']].values)
        ax2.scatter(self.data['Salary'], y_pred_all, alpha=0.6, color='purple')
        ax2.plot([self.data['Salary'].min(), self.data['Salary'].max()], 
                [self.data['Salary'].min(), self.data['Salary'].max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Salary ($)')
        ax2.set_ylabel('Predicted Salary ($)')
        ax2.set_title('Predicted vs Actual Salaries')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('polynomial_regression_results.png', dpi=300, bbox_inches='tight')
            print("✓ Plot saved as 'polynomial_regression_results.png'")
        
        plt.show()
    
    def compare_polynomial_degrees(self, max_degree=5):
        """
        Compare different polynomial degrees and find the best one.
        
        Args:
            max_degree (int): Maximum polynomial degree to test
        """
        degrees = range(1, max_degree + 1)
        train_scores = []
        test_scores = []
        
        print("\n" + "="*60)
        print("COMPARING POLYNOMIAL DEGREES")
        print("="*60)
        
        for degree in degrees:
            # Create and train model
            temp_model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            temp_model.fit(self.X_train, self.y_train)
            
            # Evaluate
            train_score = temp_model.score(self.X_train, self.y_train)
            test_score = temp_model.score(self.X_test, self.y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
            
            print(f"Degree {degree}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
        
        # Find best degree
        best_degree = degrees[np.argmax(test_scores)]
        print(f"\n✓ Best polynomial degree: {best_degree} (Test R² = {max(test_scores):.4f})")
        
        # Update model with best degree if different
        if best_degree != self.degree:
            print(f"Updating model from degree {self.degree} to degree {best_degree}")
            self.degree = best_degree
            self.train_model()
        
        return best_degree


def interactive_prediction_demo(predictor):
    """
    Interactive demo for salary prediction.
    
    Args:
        predictor: Trained PolynomialRegressionPredictor instance
    """
    print("\n" + "="*60)
    print("INTERACTIVE SALARY PREDICTION DEMO")
    print("="*60)
    print("Enter years of experience to predict salary (or 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nEnter years of experience: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            years = float(user_input)
            
            if years < 0:
                print("Error: Years of experience cannot be negative!")
                continue
            
            if years > 15:
                print("Warning: Prediction may be less accurate for experience > 15 years")
            
            predicted_salary = predictor.predict_salary(years)
            
            print(f"\n🎯 PREDICTION RESULT:")
            print(f"   Years of Experience: {years}")
            print(f"   Predicted Salary: ${predicted_salary:,.2f}")
            
            # Show similar data points from dataset
            similar_data = predictor.data[
                abs(predictor.data['YearsExperience'] - years) <= 1.0
            ].sort_values('YearsExperience')
            
            if not similar_data.empty:
                print(f"\n📊 Similar data points in dataset:")
                for _, row in similar_data.iterrows():
                    print(f"   {row['YearsExperience']} years → ${row['Salary']:,.2f}")
            
        except ValueError:
            print("Error: Please enter a valid number!")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """
    Main function to demonstrate polynomial regression salary prediction.
    """
    print("="*70)
    print("POLYNOMIAL REGRESSION SALARY PREDICTOR")
    print("="*70)
    print("This program predicts employee salaries based on years of experience")
    print("using polynomial regression with the Salary_Data.csv dataset.")
    print("="*70)
    
    # Initialize predictor
    predictor = PolynomialRegressionPredictor(degree=2)
    
    # Load data
    if not predictor.load_data('Salary_Data.csv'):
        return
    
    # Prepare data
    if not predictor.prepare_data():
        return
    
    # Compare different polynomial degrees
    best_degree = predictor.compare_polynomial_degrees()
    
    # Train model with best degree
    if not predictor.train_model():
        return
    
    # Evaluate model
    metrics = predictor.evaluate_model()
    
    # Create visualizations
    predictor.visualize_results()
    
    # Test predictions with sample values
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    sample_experiences = [2.5, 5.0, 7.5, 10.0, 12.0]
    
    for exp in sample_experiences:
        predicted = predictor.predict_salary(exp)
        print(f"Experience: {exp:4.1f} years → Predicted Salary: ${predicted:,.2f}")
    
    # Interactive demo
    interactive_prediction_demo(predictor)


if __name__ == "__main__":
    main()

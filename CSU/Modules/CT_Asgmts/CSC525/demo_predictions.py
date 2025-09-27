#!/usr/bin/env python3
"""
Simple demonstration script for polynomial regression salary predictions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def main():
    print("="*60)
    print("POLYNOMIAL REGRESSION SALARY PREDICTION DEMO")
    print("="*60)
    
    # Load data
    data = pd.read_csv('Salary_Data.csv')
    print(f"Loaded dataset with {len(data)} records")
    print(f"Experience range: {data['YearsExperience'].min():.1f} - {data['YearsExperience'].max():.1f} years")
    print(f"Salary range: ${data['Salary'].min():,.0f} - ${data['Salary'].max():,.0f}")
    
    # Prepare data
    X = data[['YearsExperience']].values
    y = data['Salary'].values
    
    # Create and train polynomial regression model (degree 3 for simplicity)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('linear', LinearRegression())
    ])
    
    model.fit(X, y)
    
    # Calculate R² score
    r2 = model.score(X, y)
    print(f"\nModel R² Score: {r2:.4f}")
    
    print("\n" + "="*60)
    print("SAMPLE SALARY PREDICTIONS")
    print("="*60)
    
    # Test predictions
    test_experiences = [1.5, 3.0, 5.0, 7.0, 9.0, 11.0]
    
    print(f"{'Experience (years)':<20} {'Predicted Salary':<20}")
    print("-" * 40)
    
    for exp in test_experiences:
        predicted = model.predict([[exp]])[0]
        print(f"{exp:<20} ${predicted:,.2f}")
    
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nEnter years of experience (or 'quit' to exit): ")
            
            if user_input.lower() in ['quit', 'q', 'exit']:
                print("Thank you for using the salary predictor!")
                break
                
            years = float(user_input)
            
            if years < 0:
                print("Error: Experience cannot be negative!")
                continue
                
            predicted_salary = model.predict([[years]])[0]
            print(f"\n🎯 Prediction: {years} years experience → ${predicted_salary:,.2f}")
            
            # Find similar data points
            similar = data[abs(data['YearsExperience'] - years) <= 1.0]
            if not similar.empty:
                print(f"\n📊 Similar records in dataset:")
                for _, row in similar.iterrows():
                    print(f"   {row['YearsExperience']} years → ${row['Salary']:,.0f}")
            
        except ValueError:
            print("Please enter a valid number!")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()

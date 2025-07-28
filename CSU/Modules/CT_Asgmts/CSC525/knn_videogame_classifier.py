#!/usr/bin/env python3
"""
KNN Video Game Preference Classifier
Predicts favorite video game genre based on age, height, weight, and gender.

Usage: python3 knn_videogame_classifier.py <age> <height> <weight> <gender>
Where:
- age: Age in years (float)
- height: Height in inches (float) 
- weight: Weight in lbs (float)
- gender: Gender (0 for female, 1 for male)

Author: Adapted from KNN Iris Classifier
Date: 2025-01-27
"""

import sys
import csv
import math

def load_data(filename):
    """
    Load video game preference data from CSV file.
    
    Args:
        filename (str): Path to the CSV file
        
    Returns:
        list: List of data points, each containing [age, height, weight, gender, genre]
    """
    data = []
    try:
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) == 5:  # age, height, weight, gender, genre
                    age = float(row[0])
                    height = float(row[1])
                    weight = float(row[2])
                    gender = float(row[3])
                    genre = row[4].strip()
                    data.append([age, height, weight, gender, genre])
        print(f"Successfully loaded {len(data)} data points from {filename}")
        return data
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        print("Please ensure video-gameData.csv is in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1 (list): First point [age, height, weight, gender]
        point2 (list): Second point [age, height, weight, gender]
        
    Returns:
        float: Euclidean distance
    """
    distance = 0
    for i in range(4):  # Only use first 4 features (age, height, weight, gender)
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

def get_neighbors(training_data, test_instance, k):
    """
    Get k nearest neighbors for a test instance.
    
    Args:
        training_data (list): Training dataset
        test_instance (list): Test instance [age, height, weight, gender]
        k (int): Number of neighbors to find
        
    Returns:
        list: k nearest neighbors with their distances
    """
    distances = []
    
    # Calculate distance to each training point
    for i, train_point in enumerate(training_data):
        dist = euclidean_distance(test_instance, train_point[:4])
        distances.append((dist, train_point, i))
    
    # Sort by distance (ascending order)
    distances.sort(key=lambda x: x[0])
    
    # Return top k neighbors
    neighbors = distances[:k]
    
    print(f"\nTop {k} nearest neighbors:")
    for i, (dist, neighbor, idx) in enumerate(neighbors):
        print(f"  {i+1}. Distance: {dist:.3f} | "
              f"Age: {neighbor[0]}, Height: {neighbor[1]}, Weight: {neighbor[2]}, "
              f"Gender: {int(neighbor[3])} | Genre: {neighbor[4]}")
    
    return neighbors

def predict_genre(neighbors):
    """
    Predict genre based on majority vote from neighbors.
    
    Args:
        neighbors (list): List of nearest neighbors
        
    Returns:
        str: Predicted genre
    """
    # Count votes for each genre
    genre_votes = {}
    
    print(f"\nVoting process:")
    for i, (dist, neighbor, idx) in enumerate(neighbors):
        genre = neighbor[4]
        if genre in genre_votes:
            genre_votes[genre] += 1
        else:
            genre_votes[genre] = 1
        print(f"  Neighbor {i+1}: {genre}")
    
    print(f"\nVote counts:")
    for genre, count in genre_votes.items():
        print(f"  {genre}: {count} vote(s)")
    
    # Find genre with most votes
    predicted_genre = max(genre_votes, key=genre_votes.get)
    max_votes = genre_votes[predicted_genre]
    
    print(f"\nWinner: {predicted_genre} with {max_votes} vote(s)")
    
    return predicted_genre

def get_genre_info(genre):
    """
    Get descriptive information about each video game genre.
    
    Args:
        genre (str): Video game genre
        
    Returns:
        str: Description of the genre
    """
    genre_info = {
        "Strategy": "Strategy games focus on skillful thinking and planning to achieve victory. "
                   "Players must use tactical and strategic thinking to outmaneuver opponents. "
                   "Examples include Chess, Civilization, StarCraft.",
        
        "RPG": "Role-Playing Games (RPG) allow players to assume the roles of characters "
               "in a fictional setting. Players control character development through "
               "leveling, skill trees, and story choices. Examples include Final Fantasy, "
               "The Elder Scrolls, Pokémon.",
        
        "Platformer": "Platform games feature jumping and climbing on suspended platforms "
                     "and obstacles. Players navigate through levels by running, jumping, "
                     "and avoiding hazards. Examples include Super Mario Bros, Sonic, "
                     "Mega Man.",
        
        "Action": "Action games emphasize physical challenges, hand-eye coordination, "
                 "and reaction time. They often feature combat, shooting, or fast-paced "
                 "gameplay. Examples include Call of Duty, Grand Theft Auto, Street Fighter."
    }
    
    return genre_info.get(genre, "Unknown genre")

def validate_input(args):
    """
    Validate command line arguments.
    
    Args:
        args (list): Command line arguments
        
    Returns:
        list: Validated input values [age, height, weight, gender]
    """
    if len(args) != 4:
        raise ValueError("Exactly 4 arguments required: age, height, weight, gender")
    
    try:
        age = float(args[0])
        height = float(args[1])
        weight = float(args[2])
        gender = float(args[3])
        
        # Validate ranges
        if age < 0 or age > 100:
            raise ValueError("Age should be between 0 and 100 years")
        if height < 30 or height > 90:
            raise ValueError("Height should be between 30 and 90 inches")
        if weight < 50 or weight > 400:
            raise ValueError("Weight should be between 50 and 400 lbs")
        if gender not in [0.0, 1.0]:
            raise ValueError("Gender should be 0 (female) or 1 (male)")
        
        return [age, height, weight, gender]
        
    except ValueError as e:
        if "could not convert" in str(e):
            raise ValueError("All arguments must be numeric values")
        else:
            raise e

def get_user_input():
    """
    Get user input interactively when no command-line arguments are provided.
    
    Returns:
        list: User input values [age, height, weight, gender]
    """
    print("\nPlease enter your information:")
    
    while True:
        try:
            age = float(input("Age (in years): "))
            if age < 0 or age > 100:
                print("Age should be between 0 and 100 years. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for age.")
    
    while True:
        try:
            height = float(input("Height (in inches): "))
            if height < 30 or height > 90:
                print("Height should be between 30 and 90 inches. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for height.")
    
    while True:
        try:
            weight = float(input("Weight (in lbs): "))
            if weight < 50 or weight > 400:
                print("Weight should be between 50 and 400 lbs. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid number for weight.")
    
    while True:
        try:
            gender_input = input("Gender (0 for female, 1 for male): ")
            gender = float(gender_input)
            if gender not in [0.0, 1.0]:
                print("Gender should be 0 (female) or 1 (male). Please try again.")
                continue
            break
        except ValueError:
            print("Please enter 0 for female or 1 for male.")
    
    return [age, height, weight, gender]

def main():
    """
    Main function implementing the 8-step KNN algorithm.
    """
    print("=" * 60)
    print("KNN Video Game Preference Classifier")
    print("=" * 60)
    
    # Check if command line arguments are provided
    if len(sys.argv) == 5:
        # Use command line arguments
        try:
            test_instance = validate_input(sys.argv[1:])
        except ValueError as e:
            print(f"\nError: {e}")
            print("Please check your input values and try again.")
            sys.exit(1)
    elif len(sys.argv) == 1:
        # Get interactive input
        try:
            test_instance = get_user_input()
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)
    else:
        # Show usage information
        print("\nUsage Option 1 (Command Line):")
        print("  python3 knn_videogame_classifier.py <age> <height> <weight> <gender>")
        print("\nUsage Option 2 (Interactive):")
        print("  python3 knn_videogame_classifier.py")
        print("  (You will be prompted for input)")
        print("\nParameters:")
        print("  age    : Age in years (e.g., 25.0)")
        print("  height : Height in inches (e.g., 68.5)")
        print("  weight : Weight in lbs (e.g., 150.0)")
        print("  gender : Gender (0 for female, 1 for male)")
        print("\nExample:")
        print("  python3 knn_videogame_classifier.py 25 68 150 1")
        print("\nAvailable genres: Strategy, RPG, Platformer, Action")
        sys.exit(1)
    
    try:
        age, height, weight, gender = test_instance
        
        print(f"\nInput Parameters:")
        print(f"  Age: {age} years")
        print(f"  Height: {height} inches")
        print(f"  Weight: {weight} lbs")
        print(f"  Gender: {'Male' if gender == 1 else 'Female'}")
        
        # Step 1: Load the data
        print(f"\nStep 1: Loading data from video-gameData.csv...")
        training_data = load_data('video-gameData.csv')
        
        # Step 2: Initialize the value of k
        k = 3
        print(f"\nStep 2: Initializing k = {k}")
        
        # Steps 3-6: Find k nearest neighbors
        print(f"\nSteps 3-6: Finding {k} nearest neighbors...")
        neighbors = get_neighbors(training_data, test_instance, k)
        
        # Steps 7-8: Predict genre based on majority vote
        print(f"\nSteps 7-8: Predicting genre based on majority vote...")
        predicted_genre = predict_genre(neighbors)
        
        # Display results
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Predicted favorite video game genre: {predicted_genre}")
        print(f"\nAbout {predicted_genre} games:")
        print(get_genre_info(predicted_genre))
        print("=" * 60)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

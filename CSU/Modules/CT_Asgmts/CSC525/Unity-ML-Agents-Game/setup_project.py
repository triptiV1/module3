#!/usr/bin/env python3
"""
Unity ML-Agents Project Setup Script
This script helps automate the setup process for the Unity ML-Agents project.
"""

import os
import subprocess
import sys

def check_requirements():
    """Check if all required tools are installed."""
    print("Checking requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if mlagents is installed
    try:
        import mlagents
        print("✅ ML-Agents package is installed")
    except ImportError:
        print("❌ ML-Agents package not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mlagents"])
        print("✅ ML-Agents package installed")
    
    return True

def create_training_script():
    """Create a training script for easy model training."""
    script_content = '''#!/bin/bash
echo "Starting ML-Agents training..."
echo "Make sure Unity project is open and playing before running this script!"
echo ""

# Train the agent
mlagents-learn config/collector_config.yaml --run-id=collector_v1 --force

echo ""
echo "Training completed! Check the 'results' folder for trained models."
echo "Copy the .onnx file to Assets/Models/ in your Unity project."
'''
    
    with open("train_agent.sh", "w") as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod("train_agent.sh", 0o755)
    print("✅ Created training script: train_agent.sh")

def create_unity_setup_guide():
    """Create a detailed Unity setup guide."""
    guide_content = '''# Unity Project Setup Guide

## Step 1: Create Unity Project
1. Open Unity Hub
2. Click "New Project"
3. Select "3D" template
4. Name: "ML-Agents-Collector-Game"
5. Location: Choose this directory
6. Click "Create project"

## Step 2: Install ML-Agents Package
1. In Unity, go to Window → Package Manager
2. Click the "+" button in top-left
3. Select "Add package from git URL"
4. Enter: `com.unity.ml-agents`
5. Click "Add"
6. Wait for installation to complete

## Step 3: Setup Scene
1. Create a new scene: File → New Scene → 3D
2. Save as "CollectorGame" in Assets/Scenes/

### Create Platform:
1. Create a Plane: GameObject → 3D Object → Plane
2. Scale to (2, 1, 2)
3. Name it "Platform"

### Create Agent:
1. Create a Cube: GameObject → 3D Object → Cube
2. Position at (0, 0.5, 0)
3. Name it "Agent"
4. Add Rigidbody component
5. Add the CollectorAgent script
6. Add Decision Requester component
7. Add Behavior Parameters component:
   - Behavior Name: "CollectorAgent"
   - Vector Observation Space Size: 14
   - Actions: Continuous, Size: 2

### Create Walls:
1. Create 4 Cubes for walls around the platform
2. Scale and position to form boundaries
3. Tag them as "Wall"

### Create Item Prefabs:
1. Create a green Cube for rewards
2. Add Collider (Is Trigger = true)
3. Tag as "Reward"
4. Save as prefab in Assets/Prefabs/
5. Repeat for red penalty cubes with "Penalty" tag

### Setup Environment Manager:
1. Create empty GameObject named "Environment"
2. Add EnvironmentManager script
3. Assign prefabs and references

### Setup UI:
1. Create Canvas
2. Add UI elements (Text, Buttons)
3. Add UIManager script

## Step 4: Training
1. Make sure scene is playing in Unity
2. Run: `./train_agent.sh`
3. Training will start - watch the console for progress

## Step 5: Testing
1. After training, copy the .onnx model to Assets/Models/
2. In Behavior Parameters, set Model to your trained model
3. Set Behavior Type to "Inference Only"
4. Play the scene to see the trained agent!
'''
    
    with open("UNITY_SETUP.md", "w") as f:
        f.write(guide_content)
    
    print("✅ Created Unity setup guide: UNITY_SETUP.md")

def main():
    """Main setup function."""
    print("🎮 Unity ML-Agents Collector Game Setup")
    print("=" * 40)
    
    if not check_requirements():
        print("❌ Requirements check failed. Please install missing dependencies.")
        return
    
    create_training_script()
    create_unity_setup_guide()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Read UNITY_SETUP.md for detailed Unity project setup")
    print("2. Open Unity Hub and create the project")
    print("3. Follow the setup guide to configure your scene")
    print("4. Run ./train_agent.sh to train your agent")
    
if __name__ == "__main__":
    main()

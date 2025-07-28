#!/usr/bin/env python3
"""
Alternative Unity ML-Agents Setup
This script provides an alternative setup that works around Python dependency conflicts.
"""

import subprocess
import sys
import os

def create_virtual_environment():
    """Create a virtual environment with compatible Python version."""
    print("Creating virtual environment for ML-Agents...")
    
    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", "ml_agents_env"], check=True)
        print("✅ Virtual environment created: ml_agents_env")
        
        # Create activation script
        activation_script = '''#!/bin/bash
# Activate ML-Agents environment
source ml_agents_env/bin/activate

# Install compatible ML-Agents version
pip install --upgrade pip
pip install mlagents==0.28.0 torch==1.8.1 --extra-index-url https://download.pytorch.org/whl/cpu

echo "ML-Agents environment activated!"
echo "Run 'mlagents-learn config/collector_config.yaml --run-id=collector_v1' to start training"
'''
        
        with open("activate_ml_env.sh", "w") as f:
            f.write(activation_script)
        
        os.chmod("activate_ml_env.sh", 0o755)
        print("✅ Created activation script: activate_ml_env.sh")
        
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False
    
    return True

def create_unity_scene_setup():
    """Create Unity scene setup instructions and prefab configurations."""
    
    scene_setup = '''# Unity Scene Setup Instructions

## 1. Create New Unity Project
1. Open Unity Hub
2. Click "New Project" 
3. Select "3D Core" template
4. Project Name: "ML-Agents-Collector-Game"
5. Location: This directory
6. Unity Version: 2022.3 LTS (recommended)

## 2. Install ML-Agents Package
1. Window → Package Manager
2. Click "+" → Add package from git URL
3. Enter: com.unity.ml-agents
4. Click "Add" and wait for installation

## 3. Scene Setup

### Create Main Scene:
- File → New Scene → 3D
- Save as "CollectorGame" in Assets/Scenes/

### Platform Setup:
```
GameObject → 3D Object → Plane
- Name: "Platform"
- Transform: Position (0, 0, 0), Scale (2, 1, 2)
- Material: Create new material, set to white/gray
```

### Agent Setup:
```
GameObject → 3D Object → Cube
- Name: "Agent" 
- Transform: Position (0, 0.5, 0), Scale (1, 1, 1)
- Material: Create blue material
- Add Component: Rigidbody
- Add Component: CollectorAgent (our script)
- Add Component: Decision Requester
  - Decision Period: 5
- Add Component: Behavior Parameters
  - Behavior Name: "CollectorAgent"
  - Behavior Type: "Default"
  - Vector Observation Space Size: 14
  - Actions: Continuous, Size: 2
```

### Environment Boundaries:
Create 4 walls around the platform:
```
GameObject → 3D Object → Cube (repeat 4 times)
- Names: "Wall_North", "Wall_South", "Wall_East", "Wall_West"
- Scale: (10, 2, 1) for North/South, (1, 2, 10) for East/West
- Positions: 
  - North: (0, 1, 5)
  - South: (0, 1, -5)  
  - East: (5, 1, 0)
  - West: (-5, 1, 0)
- Tag: "Wall"
```

### Reward Items (Green Cubes):
```
GameObject → 3D Object → Cube
- Name: "RewardItem"
- Scale: (0.5, 0.5, 0.5)
- Material: Create green material
- Add Component: Box Collider
  - Is Trigger: ✓
- Tag: "Reward"
- Save as Prefab in Assets/Prefabs/
```

### Penalty Items (Red Cubes):
```
GameObject → 3D Object → Cube  
- Name: "PenaltyItem"
- Scale: (0.5, 0.5, 0.5)
- Material: Create red material
- Add Component: Box Collider
  - Is Trigger: ✓
- Tag: "Penalty"
- Save as Prefab in Assets/Prefabs/
```

### Environment Manager:
```
GameObject → Create Empty
- Name: "EnvironmentManager"
- Add Component: EnvironmentManager (our script)
- Assign prefabs and references in inspector
```

### UI Setup:
```
GameObject → UI → Canvas
- Add UI elements for score, episode count, etc.
- Add UIManager script to Canvas
```

## 4. Script Assignment
Make sure all scripts are properly assigned:
- CollectorAgent script on Agent GameObject
- EnvironmentManager script on EnvironmentManager GameObject  
- UIManager script on Canvas

## 5. Training Setup
After scene is complete:
1. Run `./activate_ml_env.sh` to set up Python environment
2. Start Unity and play the scene
3. In terminal: `mlagents-learn config/collector_config.yaml --run-id=collector_v1`
4. Press Play in Unity when prompted

## 6. Testing
- Use arrow keys for manual control (heuristic mode)
- Press Space to reset environment
- Watch agent learn to collect green cubes and avoid red ones
'''
    
    with open("UNITY_SCENE_SETUP.md", "w") as f:
        f.write(scene_setup)
    
    print("✅ Created detailed Unity scene setup guide")

def create_troubleshooting_guide():
    """Create troubleshooting guide for common issues."""
    
    troubleshooting = '''# Troubleshooting Guide

## Python Environment Issues

### Problem: ML-Agents installation fails
**Solution:**
```bash
# Use virtual environment
python3 -m venv ml_agents_env
source ml_agents_env/bin/activate
pip install --upgrade pip
pip install mlagents==0.28.0
```

### Problem: Torch compatibility issues
**Solution:**
```bash
# Install specific torch version
pip install torch==1.8.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

## Unity Issues

### Problem: ML-Agents package not found
**Solution:**
1. Check Unity version (2022.3 LTS recommended)
2. Use Package Manager → Add from Git URL
3. URL: `com.unity.ml-agents`

### Problem: Agent not learning
**Solution:**
1. Check Behavior Parameters settings
2. Verify Decision Requester is added
3. Ensure training config matches behavior name
4. Check reward function in CollectorAgent script

### Problem: Training connection fails
**Solution:**
1. Start Unity scene first
2. Press Play button
3. Then run mlagents-learn command
4. Unity should show "Connected to trainer"

## Performance Issues

### Problem: Training is slow
**Solution:**
1. Reduce Time Scale in Unity (Edit → Project Settings → Time)
2. Increase Decision Period in Decision Requester
3. Use multiple training environments

### Problem: Agent behavior is erratic
**Solution:**
1. Check reward values (not too high/low)
2. Verify observation space size matches script
3. Adjust training hyperparameters in config file

## Alternative Training Methods

If Python setup continues to fail, you can:
1. Use Unity's built-in ML-Agents examples
2. Download pre-trained models from Unity ML-Agents repository
3. Focus on Unity scene setup and manual testing
4. Use heuristic mode for demonstration

## Getting Help
- Unity ML-Agents Documentation: https://github.com/Unity-Technologies/ml-agents
- Unity Forums: https://forum.unity.com/
- ML-Agents Discord: Check Unity's official channels
'''
    
    with open("TROUBLESHOOTING.md", "w") as f:
        f.write(troubleshooting)
    
    print("✅ Created troubleshooting guide")

def main():
    """Main setup function."""
    print("🔧 Alternative Unity ML-Agents Setup")
    print("=" * 40)
    
    create_virtual_environment()
    create_unity_scene_setup()
    create_troubleshooting_guide()
    
    print("\n🎉 Alternative setup completed!")
    print("\nNext steps:")
    print("1. Read UNITY_SCENE_SETUP.md for detailed Unity configuration")
    print("2. Run ./activate_ml_env.sh to set up Python environment")
    print("3. Open Unity Hub and create the project")
    print("4. Follow the scene setup guide step by step")
    print("5. If issues persist, check TROUBLESHOOTING.md")

if __name__ == "__main__":
    main()

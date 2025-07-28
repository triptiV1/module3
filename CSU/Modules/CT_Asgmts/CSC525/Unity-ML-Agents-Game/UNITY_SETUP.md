# Unity Project Setup Guide

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

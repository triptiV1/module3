# Unity ML-Agents Collector Game

This project demonstrates a Unity game using ML-Agents where an intelligent agent learns to collect items while avoiding obstacles.

## Project Structure
- `Assets/Scripts/` - C# scripts for the game and ML-Agent
- `Assets/Scenes/` - Unity scenes
- `Assets/Prefabs/` - Game object prefabs
- `config/` - ML-Agents training configuration
- `models/` - Trained models will be saved here

## Game Description
The agent (a blue cube) learns to:
- Navigate in a 3D environment
- Collect green reward items (+1 point each)
- Avoid red penalty items (-1 point each)
- Stay within the platform boundaries

## Setup Instructions

### 1. Unity Setup
1. Open Unity Hub
2. Create a new 3D project named "ML-Agents-Collector-Game"
3. Install ML-Agents package via Package Manager:
   - Window → Package Manager
   - Click "+" → Add package from git URL
   - Enter: `com.unity.ml-agents`

### 2. Python Environment
```bash
pip3 install mlagents
```

### 3. Training the Agent
```bash
mlagents-learn config/collector_config.yaml --run-id=collector_v1
```

### 4. Running with Trained Model
After training, the model will be automatically loaded and the agent will demonstrate learned behavior.

## Controls
- Space: Reset environment
- R: Toggle between training and inference mode

## Technical Details
- **Observation Space**: Agent's position, velocity, and raycasts to detect items
- **Action Space**: Continuous movement in X and Z directions
- **Reward Function**: +1 for collecting green items, -1 for red items, small negative for time
- **Training Algorithm**: PPO (Proximal Policy Optimization)

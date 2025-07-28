# Unity Scene Setup Instructions

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

# Troubleshooting Guide

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

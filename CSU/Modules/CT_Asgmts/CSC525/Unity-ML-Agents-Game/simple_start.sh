#!/bin/bash

echo "🎮 Unity ML-Agents Simple Start Guide"
echo "====================================="
echo ""

echo "Step 1: Check if Unity Hub is installed..."
if [ -d "/Applications/Unity Hub.app" ]; then
    echo "✅ Unity Hub found!"
    echo "Opening Unity Hub..."
    open "/Applications/Unity Hub.app"
else
    echo "❌ Unity Hub not found. Please install Unity Hub first."
    echo "Download from: https://unity3d.com/get-unity/download"
    exit 1
fi

echo ""
echo "Step 2: Unity Project Setup Instructions"
echo "----------------------------------------"
echo "In Unity Hub:"
echo "1. Click 'New Project'"
echo "2. Select '3D Core' template"
echo "3. Project Name: ML-Agents-Collector-Game"
echo "4. Location: $(pwd)"
echo "5. Click 'Create project'"
echo ""

echo "Step 3: Install ML-Agents Package"
echo "---------------------------------"
echo "In Unity Editor:"
echo "1. Window → Package Manager"
echo "2. Click '+' → Add package from git URL"
echo "3. Enter: com.unity.ml-agents"
echo "4. Click 'Add'"
echo ""

echo "Step 4: Scene Setup"
echo "------------------"
echo "Follow the detailed instructions in:"
echo "📖 UNITY_SCENE_SETUP.md"
echo ""

echo "Step 5: For Training (Optional)"
echo "------------------------------"
echo "If you want to train the AI agent:"
echo "1. Try installing ML-Agents with: pip3 install mlagents"
echo "2. Or use the virtual environment approach"
echo "3. Or focus on Unity scene setup and manual testing first"
echo ""

echo "🚀 Ready to start! Unity Hub should be opening now."
echo "📖 Read UNITY_SCENE_SETUP.md for detailed scene configuration."

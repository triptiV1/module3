#!/bin/bash
# Virtual Environment Setup Script for NLP Chatbot

echo "🚀 Setting up Virtual Environment for NLP Chatbot..."

# Create virtual environment
python3 -m venv chatbot_env

# Activate virtual environment
source chatbot_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "source chatbot_env/bin/activate"
echo ""
echo "To run the chatbot:"
echo "python3 final_nlp_chatbot.py"
echo ""
echo "To deactivate when done:"
echo "deactivate"

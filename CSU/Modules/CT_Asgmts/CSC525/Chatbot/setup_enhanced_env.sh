#!/bin/bash
# Setup script for Enhanced NLP Chatbot Environment

echo "🚀 Setting up Enhanced NLP Chatbot Environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "chatbot_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv chatbot_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source chatbot_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "📖 Downloading NLTK data..."
python3 -c "
import nltk
print('Downloading NLTK data...')
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
print('✅ NLTK data downloaded successfully')
"

# Download spaCy model
echo "🧠 Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm

echo ""
echo "✅ Enhanced NLP Chatbot environment setup complete!"
echo ""
echo "🎯 To run the chatbot:"
echo "   1. Console version: python3 enhanced_nlp_chatbot.py"
echo "   2. Web interface: python3 web_interface.py"
echo ""
echo "🌐 Web interface will be available at: http://localhost:5000"
echo ""
echo "💡 Features available:"
echo "   • Advanced NER with spaCy"
echo "   • POS tagging and linguistic analysis"
echo "   • Ambiguity detection and clarification"
echo "   • Fuzzy matching for better understanding"
echo "   • Scalable JSON knowledge base"
echo "   • Enhanced sentiment analysis with VADER"
echo "   • Web API for platform integration"

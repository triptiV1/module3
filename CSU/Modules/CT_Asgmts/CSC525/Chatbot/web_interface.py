#!/usr/bin/env python3
"""
Web Interface for Enhanced NLP Chatbot
Provides REST API and web interface for platform integration
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os
from enhanced_nlp_chatbot import EnhancedNLPChatbot

app = Flask(__name__)
CORS(app)

# Global chatbot instance
chatbot = None

def initialize_chatbot():
    """Initialize the chatbot instance"""
    global chatbot
    if chatbot is None:
        try:
            chatbot = EnhancedNLPChatbot()
            print("✅ Chatbot initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            return False
    return True

@app.route('/')
def index():
    """Main web interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Chat API endpoint"""
    try:
        if not initialize_chatbot():
            return jsonify({'error': 'Chatbot initialization failed'}), 500
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get chatbot response
        response = chatbot.generate_response(user_message)
        
        # Extract additional analysis
        entities = chatbot.extract_entities_advanced(user_message)
        pos_tags = chatbot.get_pos_tags(user_message)
        intent, intent_confidence = chatbot.classify_intent_enhanced(user_message)
        sentiment, sentiment_confidence = chatbot.analyze_sentiment_enhanced(user_message)
        
        return jsonify({
            'response': response,
            'analysis': {
                'intent': intent,
                'intent_confidence': round(intent_confidence, 3),
                'sentiment': sentiment,
                'sentiment_confidence': round(sentiment_confidence, 3),
                'entities': entities,
                'pos_tags': pos_tags[:10]  # Limit for readability
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'chatbot_ready': chatbot is not None})

@app.route('/api/stats')
def get_stats():
    """Get chatbot statistics"""
    try:
        if not initialize_chatbot():
            return jsonify({'error': 'Chatbot not initialized'}), 500
        
        return jsonify({
            'conversation_count': len(chatbot.conversation_history),
            'knowledge_base_size': len(chatbot.knowledge_base.get('categories', {})),
            'user_preferences': dict(chatbot.user_preferences)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Create templates directory and HTML template
def create_templates():
    """Create HTML template for web interface"""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced NLP Chatbot</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .chat-container {
            display: flex;
            height: 600px;
        }
        
        .chat-area {
            flex: 2;
            display: flex;
            flex-direction: column;
        }
        
        .analysis-panel {
            flex: 1;
            background: #f8f9fa;
            border-left: 1px solid #dee2e6;
            padding: 20px;
            overflow-y: auto;
        }
        
        .messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #fafafa;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .bot-message {
            background: #e9ecef;
            color: #333;
        }
        
        .input-area {
            padding: 20px;
            border-top: 1px solid #dee2e6;
            background: white;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        .message-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #dee2e6;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }
        
        .message-input:focus {
            border-color: #007bff;
        }
        
        .send-button {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        
        .send-button:hover {
            background: #0056b3;
        }
        
        .analysis-section {
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 10px;
            border-left: 4px solid #007bff;
        }
        
        .analysis-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #007bff;
        }
        
        .entity-tag {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin: 2px;
        }
        
        .confidence-bar {
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
            transition: width 0.3s;
        }
        
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced NLP Chatbot</h1>
            <p>Advanced Natural Language Understanding with spaCy & NLTK</p>
        </div>
        
        <div class="chat-container">
            <div class="chat-area">
                <div class="messages" id="messages">
                    <div class="message bot-message">
                        Hello! I'm an enhanced NLP chatbot with advanced understanding capabilities. 
                        I can analyze your messages for intent, sentiment, entities, and more. 
                        Try asking me something ambiguous like "Tell me about Python" and I'll ask for clarification!
                    </div>
                </div>
                
                <div class="input-area">
                    <div class="input-group">
                        <input type="text" class="message-input" id="messageInput" 
                               placeholder="Type your message here..." maxlength="500">
                        <button class="send-button" id="sendButton" onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
            
            <div class="analysis-panel" id="analysisPanel">
                <h3>💡 Analysis Panel</h3>
                <p>Send a message to see detailed NLP analysis including intent classification, sentiment analysis, named entity recognition, and part-of-speech tagging.</p>
            </div>
        </div>
    </div>

    <script>
        let isLoading = false;
        
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        async function sendMessage() {
            if (isLoading) return;
            
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            // Set loading state
            setLoading(true);
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage('Error: ' + data.error, 'bot');
                } else {
                    addMessage(data.response, 'bot');
                    updateAnalysis(data.analysis);
                }
                
            } catch (error) {
                addMessage('Connection error. Please try again.', 'bot');
                console.error('Error:', error);
            } finally {
                setLoading(false);
            }
        }
        
        function addMessage(text, sender) {
            const messages = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = text;
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function updateAnalysis(analysis) {
            const panel = document.getElementById('analysisPanel');
            
            panel.innerHTML = `
                <h3>💡 Analysis Results</h3>
                
                <div class="analysis-section">
                    <div class="analysis-title">Intent Classification</div>
                    <div><strong>${analysis.intent}</strong></div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${analysis.intent_confidence * 100}%"></div>
                    </div>
                    <small>Confidence: ${(analysis.intent_confidence * 100).toFixed(1)}%</small>
                </div>
                
                <div class="analysis-section">
                    <div class="analysis-title">Sentiment Analysis</div>
                    <div><strong>${analysis.sentiment}</strong></div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${analysis.sentiment_confidence * 100}%"></div>
                    </div>
                    <small>Confidence: ${(analysis.sentiment_confidence * 100).toFixed(1)}%</small>
                </div>
                
                <div class="analysis-section">
                    <div class="analysis-title">Named Entities</div>
                    <div>
                        ${Object.keys(analysis.entities).length > 0 
                            ? Object.entries(analysis.entities).map(([type, entities]) => 
                                `<div><strong>${type}:</strong> ${entities.map(e => `<span class="entity-tag">${e}</span>`).join('')}</div>`
                              ).join('')
                            : '<em>No entities detected</em>'
                        }
                    </div>
                </div>
                
                <div class="analysis-section">
                    <div class="analysis-title">Part-of-Speech Tags</div>
                    <div>
                        ${analysis.pos_tags.map(([word, pos]) => 
                            `<span class="entity-tag">${word} (${pos})</span>`
                        ).join(' ')}
                    </div>
                </div>
            `;
        }
        
        function setLoading(loading) {
            isLoading = loading;
            const container = document.querySelector('.container');
            const button = document.getElementById('sendButton');
            
            if (loading) {
                container.classList.add('loading');
                button.textContent = 'Sending...';
            } else {
                container.classList.remove('loading');
                button.textContent = 'Send';
            }
        }
    </script>
</body>
</html>'''
    
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    print("Setting up web interface...")
    create_templates()
    print("✅ Web interface ready!")
    print("\n🌐 Starting Flask server...")
    print("📱 Access the chatbot at: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/api/chat")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

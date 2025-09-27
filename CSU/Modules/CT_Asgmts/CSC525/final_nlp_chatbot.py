#!/usr/bin/env python3
"""
Final NLP Chatbot Project - CSC525
Advanced Conversational AI with Multiple NLP Learning Methods

This chatbot implements multiple NLP techniques including:
- Intent classification using neural networks
- Named Entity Recognition (NER)
- Sentiment analysis
- Context-aware response generation
- Learning from conversation history

Author: Tripti Vishwakarma
Course: CSC525 - Machine Learning
"""

import os
import json
import re
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class AdvancedNLPChatbot:
    """
    Advanced NLP Chatbot with multiple learning methods:
    1. Intent Classification using Neural Networks
    2. Named Entity Recognition
    3. Sentiment Analysis
    4. Context-aware Response Generation
    5. Conversation Learning and Adaptation
    """
    
    def __init__(self):
        # Basic stopwords for text processing
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        # Intent classification components
        self.intent_model = None
        self.intent_tokenizer = None
        self.intent_vectorizer = None
        self.intent_classifier = None
        
        # Sentiment analysis
        self.sentiment_vectorizer = None
        self.sentiment_classifier = None
        
        # Conversation context and learning
        self.conversation_history = []
        self.user_preferences = defaultdict(int)
        self.context_memory = []
        self.response_templates = {}
        
        # NER patterns - more restrictive to avoid false matches
        self.ner_patterns = {
            'PERSON': r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b',
            'TIME': r'\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)\b',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\btoday\b|\btomorrow\b|\byesterday\b',
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        # Initialize training data and models
        self._initialize_training_data()
        self._train_models()
        
    def _initialize_training_data(self):
        """Initialize comprehensive training data for the chatbot"""
        
        # Intent classification training data
        self.intent_data = {
            'greeting': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                'howdy', 'greetings', 'what\'s up', 'how are you', 'nice to meet you'
            ],
            'farewell': [
                'goodbye', 'bye', 'see you later', 'farewell', 'take care',
                'catch you later', 'until next time', 'have a good day', 'talk to you soon'
            ],
            'question': [
                'what is', 'how do', 'can you explain', 'tell me about', 'what does',
                'how does', 'why is', 'when did', 'where is', 'who is', 'help me understand'
            ],
            'help': [
                'help me', 'I need help', 'can you help', 'assist me', 'support',
                'I don\'t understand', 'confused', 'explain please', 'guide me'
            ],
            'compliment': [
                'you are great', 'awesome', 'fantastic', 'wonderful', 'amazing',
                'excellent work', 'well done', 'impressive', 'brilliant', 'outstanding'
            ],
            'weather': [
                'weather', 'temperature', 'rain', 'sunny', 'cloudy', 'forecast',
                'hot', 'cold', 'storm', 'snow', 'humidity', 'climate'
            ],
            'technology': [
                'computer', 'software', 'programming', 'artificial intelligence', 'machine learning',
                'technology', 'internet', 'website', 'app', 'digital', 'code', 'algorithm'
            ],
            'personal': [
                'my name is', 'I am', 'I like', 'I love', 'I hate', 'I prefer',
                'my favorite', 'about me', 'personally', 'I think', 'I believe'
            ],
            'education': [
                'school', 'university', 'college', 'study', 'learn', 'education',
                'course', 'class', 'teacher', 'student', 'homework', 'assignment'
            ],
            'small_talk': [
                'how was your day', 'what do you think', 'interesting', 'really',
                'that\'s cool', 'nice', 'okay', 'sure', 'maybe', 'probably'
            ]
        }
        
        # Response templates for each intent
        self.response_templates = {
            'greeting': [
                "Hello! I'm an advanced NLP chatbot. How can I help you today?",
                "Hi there! Great to chat with you. What would you like to discuss?",
                "Greetings! I'm here to assist you with any questions or conversations.",
                "Hello! I'm ready to help. What's on your mind?"
            ],
            'farewell': [
                "Goodbye! It was great chatting with you. Take care!",
                "See you later! Thanks for the interesting conversation.",
                "Farewell! Feel free to come back anytime for more chat.",
                "Bye! Hope to talk with you again soon."
            ],
            'question': [
                "That's an interesting question! Let me think about that...",
                "Great question! Based on my knowledge, I'd say...",
                "I'd be happy to help explain that. Here's what I know...",
                "Excellent inquiry! Let me provide you with some information..."
            ],
            'help': [
                "I'm here to help! Can you tell me more specifically what you need assistance with?",
                "Of course I can help! What particular topic or issue are you dealing with?",
                "I'd be glad to assist you. Could you provide more details about what you need?",
                "Help is on the way! What specific area would you like me to focus on?"
            ],
            'compliment': [
                "Thank you so much! I appreciate your kind words.",
                "That's very nice of you to say! I'm glad I could help.",
                "Thanks! I try my best to be helpful and engaging.",
                "I appreciate the compliment! It motivates me to keep improving."
            ],
            'weather': [
                "I don't have access to real-time weather data, but I'd be happy to discuss weather patterns or climate topics!",
                "Weather is fascinating! Are you asking about current conditions or weather in general?",
                "I can't check the current weather, but I can discuss meteorology and weather phenomena!",
                "Weather-related questions are interesting! What specific aspect would you like to explore?"
            ],
            'technology': [
                "Technology is one of my favorite topics! What specific area interests you?",
                "Great choice of topic! Technology is rapidly evolving. What would you like to know?",
                "I love discussing technology! From AI to programming, there's so much to explore.",
                "Technology fascinates me! Which aspect would you like to dive into?"
            ],
            'personal': [
                "Thanks for sharing! I find personal experiences and preferences really interesting.",
                "That's great to know about you! Personal connections make conversations more meaningful.",
                "I appreciate you telling me about yourself. It helps me understand you better.",
                "Thanks for the personal insight! It's nice to learn more about who I'm talking with."
            ],
            'education': [
                "Education is so important! What area of learning are you interested in?",
                "Learning never stops! What educational topic would you like to explore?",
                "Education opens so many doors. What specific subject or level are you thinking about?",
                "Great topic! Education and learning are fundamental to growth. What's your focus?"
            ],
            'small_talk': [
                "Absolutely! I enjoy these casual conversations.",
                "That's a nice way to put it! I like chatting about everyday things too.",
                "Indeed! Sometimes the best conversations are just friendly exchanges.",
                "I agree! These informal chats can be really engaging."
            ]
        }
        
        # Sentiment training data
        self.sentiment_data = {
            'positive': [
                'I love this', 'amazing', 'wonderful', 'great job', 'fantastic',
                'excellent', 'awesome', 'brilliant', 'perfect', 'outstanding',
                'happy', 'excited', 'thrilled', 'delighted', 'pleased',
                'I am happy', 'feeling good', 'great day', 'love it', 'so good',
                'wonderful time', 'excellent work', 'very happy', 'feeling great'
            ],
            'negative': [
                'I hate this', 'terrible', 'awful', 'horrible', 'disgusting',
                'worst', 'bad', 'disappointing', 'frustrated', 'angry',
                'sad', 'upset', 'annoyed', 'irritated', 'dissatisfied',
                'I am not happy', 'not good', 'feeling bad', 'hate it', 'so bad',
                'terrible time', 'poor work', 'very sad', 'feeling terrible',
                'not satisfied', 'unhappy with', 'disappointed with'
            ],
            'neutral': [
                'okay', 'fine', 'average', 'normal', 'standard', 'typical',
                'regular', 'ordinary', 'common', 'usual', 'moderate',
                'acceptable', 'reasonable', 'fair', 'adequate', 'how are you',
                'what is', 'can you', 'tell me', 'I think', 'maybe'
            ]
        }
    
    def _train_models(self):
        """Train all NLP models"""
        print("Training NLP models...")
        
        # Train intent classification model
        self._train_intent_classifier()
        
        # Train sentiment analysis model
        self._train_sentiment_classifier()
        
        print("All models trained successfully!")
    
    def _train_intent_classifier(self):
        """Train neural network for intent classification"""
        
        # Prepare training data
        texts = []
        labels = []
        label_to_idx = {}
        idx_to_label = {}
        
        for idx, (intent, examples) in enumerate(self.intent_data.items()):
            label_to_idx[intent] = idx
            idx_to_label[idx] = intent
            for example in examples:
                texts.append(example)
                labels.append(idx)
        
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        
        # Tokenize and vectorize
        self.intent_tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        self.intent_tokenizer.fit_on_texts(texts)
        
        sequences = self.intent_tokenizer.texts_to_sequences(texts)
        max_length = 20
        X = pad_sequences(sequences, maxlen=max_length)
        y = to_categorical(labels, num_classes=len(self.intent_data))
        
        # Build neural network model
        model = Sequential([
            Embedding(5000, 64, input_length=max_length),
            Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(self.intent_data), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        model.fit(X, y, epochs=50, batch_size=8, verbose=0)
        self.intent_model = model
        self.max_length = max_length
    
    def _train_sentiment_classifier(self):
        """Train sentiment analysis classifier"""
        
        # Prepare sentiment training data
        texts = []
        labels = []
        
        for sentiment, examples in self.sentiment_data.items():
            for example in examples:
                texts.append(example)
                labels.append(sentiment)
        
        # Vectorize text
        self.sentiment_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = self.sentiment_vectorizer.fit_transform(texts)
        
        # Train classifier
        self.sentiment_classifier = LogisticRegression()
        self.sentiment_classifier.fit(X, labels)
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\?\!\.]', '', text)
        
        # Simple tokenization
        tokens = text.split()
        
        # Remove stopwords and short tokens
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def extract_entities(self, text):
        """Named Entity Recognition using pattern matching"""
        entities = {}
        
        for entity_type, pattern in self.ner_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def classify_intent(self, text):
        """Classify user intent using neural network"""
        processed_text = self.preprocess_text(text)
        sequence = self.intent_tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=self.max_length)
        
        prediction = self.intent_model.predict(padded, verbose=0)
        intent_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][intent_idx])
        
        intent = self.idx_to_label[intent_idx]
        return intent, confidence
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of user input with keyword-based fallback"""
        # First try keyword-based detection for common patterns
        text_lower = text.lower()
        
        # Strong negative indicators
        if any(phrase in text_lower for phrase in ['not happy', 'hate', 'terrible', 'awful', 'frustrated', 'angry', 'disappointed', 'not good']):
            return 'negative', 0.9
        
        # Strong positive indicators  
        if any(phrase in text_lower for phrase in ['i am happy', 'love', 'great', 'wonderful', 'excellent', 'amazing']):
            return 'positive', 0.9
            
        # Use ML classifier for other cases
        try:
            processed_text = self.preprocess_text(text)
            if not processed_text.strip():
                return 'neutral', 0.5
                
            vectorized = self.sentiment_vectorizer.transform([processed_text])
            sentiment = self.sentiment_classifier.predict(vectorized)[0]
            probabilities = self.sentiment_classifier.predict_proba(vectorized)[0]
            confidence = max(probabilities)
            
            return sentiment, confidence
        except:
            return 'neutral', 0.5
    
    def update_context(self, user_input, bot_response):
        """Update conversation context for learning"""
        context_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'entities': self.extract_entities(user_input),
            'intent': self.classify_intent(user_input)[0],
            'sentiment': self.analyze_sentiment(user_input)[0]
        }
        
        self.context_memory.append(context_entry)
        
        # Keep only last 10 exchanges for context
        if len(self.context_memory) > 10:
            self.context_memory.pop(0)
    
    def learn_from_conversation(self, user_input):
        """Learn user preferences and patterns"""
        # Extract keywords and update preferences
        processed = self.preprocess_text(user_input)
        tokens = processed.split()
        
        for token in tokens:
            self.user_preferences[token] += 1
    
    def generate_contextual_response(self, intent, user_input, sentiment):
        """Generate context-aware response"""
        base_responses = self.response_templates.get(intent, [
            "That's interesting! Tell me more.",
            "I understand. Can you elaborate on that?",
            "Thanks for sharing that with me.",
            "I see what you mean. What else would you like to discuss?"
        ])
        
        # Select base response
        response = np.random.choice(base_responses)
        
        # Modify response based on sentiment
        if sentiment == 'negative':
            if 'not happy' in user_input.lower() or 'frustrated' in user_input.lower() or 'bad' in user_input.lower():
                response = "I'm sorry to hear you're having difficulties. " + response
            else:
                response = "I sense some concern in your message. " + response
        elif sentiment == 'positive':
            if 'happy' in user_input.lower() or 'good' in user_input.lower() or 'great' in user_input.lower():
                response = "I can tell you're in a good mood! " + response
        
        return response
    
    def get_response(self, user_input):
        """Main method to get chatbot response"""
        if not user_input.strip():
            return "I didn't catch that. Could you please say something?"
        
        # Learn from this interaction
        self.learn_from_conversation(user_input)
        
        # Classify intent and analyze sentiment
        intent, intent_confidence = self.classify_intent(user_input)
        sentiment, sentiment_confidence = self.analyze_sentiment(user_input)
        
        # Generate response
        response = self.generate_contextual_response(intent, user_input, sentiment)
        
        # Update conversation context
        self.update_context(user_input, response)
        
        # Add learning indicator for high-confidence predictions
        if intent_confidence > 0.8:
            response += f" (I'm {intent_confidence:.1%} confident this is about {intent})"
        
        return response
    
    def get_conversation_stats(self):
        """Get statistics about the conversation"""
        if not self.context_memory:
            return "No conversation history yet."
        
        intents = [entry['intent'] for entry in self.context_memory]
        sentiments = [entry['sentiment'] for entry in self.context_memory]
        
        intent_counts = Counter(intents)
        sentiment_counts = Counter(sentiments)
        
        stats = f"""
Conversation Statistics:
- Total exchanges: {len(self.context_memory)}
- Most common intent: {intent_counts.most_common(1)[0][0]} ({intent_counts.most_common(1)[0][1]} times)
- Overall sentiment: {sentiment_counts.most_common(1)[0][0]}
- Top user preferences: {', '.join([word for word, count in Counter(self.user_preferences).most_common(3)])}
        """
        
        return stats.strip()

def main():
    """Main chatbot interface"""
    print("=" * 60)
    print("🤖 ADVANCED NLP CHATBOT - CSC525 Final Project")
    print("=" * 60)
    print("Initializing advanced NLP models...")
    
    # Initialize chatbot
    chatbot = AdvancedNLPChatbot()
    
    print("\n✅ Chatbot ready! Features include:")
    print("   • Intent Classification (Neural Network)")
    print("   • Sentiment Analysis")
    print("   • Named Entity Recognition")
    print("   • Context-Aware Responses")
    print("   • Conversation Learning")
    print("\n💡 Commands:")
    print("   • Type 'stats' to see conversation statistics")
    print("   • Type 'quit' or 'exit' to end the conversation")
    print("   • Type anything else to chat!")
    print("\n" + "=" * 60)
    
    conversation_count = 0
    
    while True:
        try:
            user_input = input(f"\n[{conversation_count + 1}] You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print(f"\n🤖 Bot: {chatbot.get_response(user_input)}")
                print("\nThank you for testing the Advanced NLP Chatbot!")
                print("This chatbot demonstrated:")
                print("✓ Neural network-based intent classification")
                print("✓ Machine learning sentiment analysis")
                print("✓ Named entity recognition")
                print("✓ Context-aware response generation")
                print("✓ Conversation learning and adaptation")
                break
            
            elif user_input.lower() == 'stats':
                print(f"\n📊 {chatbot.get_conversation_stats()}")
                continue
            
            elif not user_input:
                print("\n🤖 Bot: Please type something to continue our conversation!")
                continue
            
            # Get chatbot response
            response = chatbot.get_response(user_input)
            print(f"\n🤖 Bot: {response}")
            
            conversation_count += 1
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for chatting with the Advanced NLP Chatbot!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()

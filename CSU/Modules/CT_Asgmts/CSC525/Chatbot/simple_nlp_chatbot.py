#!/usr/bin/env python3
"""
Simple NLP Chatbot using NLTK, ChatterBot, TextBlob, and spaCy
Created fresh and clean for reliable operation
"""

import os
import warnings
import ssl
from datetime import datetime
warnings.filterwarnings('ignore')

# Fix SSL issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def setup_nltk():
    """Setup NLTK data"""
    import nltk
    
    required_data = [
        'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 
        'stopwords', 'vader_lexicon', 'wordnet'
    ]
    
    for item in required_data:
        try:
            nltk.download(item, quiet=True)
        except:
            print(f"⚠️ Could not download {item}, but continuing...")

class SimpleNLPChatbot:
    """Simple NLP Chatbot with all 4 requested libraries"""
    
    def __init__(self):
        print("🤖 Setting up Simple NLP Chatbot...")
        
        # Setup NLTK
        setup_nltk()
        
        # Import libraries
        self._import_libraries()
        
        # Initialize components
        self._setup_components()
        
        # Comprehensive knowledge base with specific topics
        self.knowledge = {
            # Python Programming - Specific concepts
            'generator function in python': "A generator function in Python uses 'yield' instead of 'return' to produce a series of values lazily. It creates an iterator that generates values on-demand, saving memory. Example: def count_up(): i = 1; while True: yield i; i += 1",
            'python generator': "A generator in Python is an iterator that generates values on-demand using yield. It's memory efficient for large datasets since values are produced one at a time rather than storing everything in memory.",
            'list comprehension': "List comprehension in Python is a concise way to create lists. Syntax: [expression for item in iterable if condition]. Example: [x*2 for x in range(5)] creates [0, 2, 4, 6, 8].",
            'python decorator': "A Python decorator is a function that modifies or extends another function's behavior without changing its code. Decorators use @decorator_name syntax above function definitions.",
            'lambda function': "Lambda functions in Python are small anonymous functions defined with 'lambda' keyword. Syntax: lambda arguments: expression. Example: lambda x: x*2",
            
            # General Programming
            'python': "Python is a high-level, interpreted programming language known for its simple syntax and readability. It's widely used in web development, data science, AI, and automation.",
            'programming': "Programming is the process of designing and building computer programs by writing code in programming languages to solve problems or create applications.",
            
            # Machine Learning & AI
            'machine learning': "Machine learning is a subset of AI where computers learn patterns from data to make predictions or decisions without being explicitly programmed for every specific task.",
            'artificial intelligence': "Artificial Intelligence (AI) is technology that enables machines to simulate human intelligence, including learning, reasoning, and problem-solving.",
            'neural network': "A neural network is a computing system inspired by biological neural networks. It consists of interconnected nodes (neurons) that process information and learn patterns.",
            'deep learning': "Deep learning uses neural networks with multiple layers (deep networks) to learn complex patterns in large amounts of data, especially effective for images, speech, and text.",
            
            # NLP Libraries
            'nltk': "NLTK (Natural Language Toolkit) is a comprehensive Python library for natural language processing with tools for tokenization, parsing, classification, and linguistic analysis.",
            'spacy': "spaCy is an industrial-strength NLP library for Python designed for production use. It provides fast, accurate linguistic annotations including tokenization, POS tagging, and NER.",
            'textblob': "TextBlob is a Python library that provides a simple API for common NLP tasks like sentiment analysis, noun phrase extraction, and spelling correction.",
            'chatterbot': "ChatterBot is a Python library for creating conversational AI. It uses machine learning algorithms to generate responses based on training conversations.",
            
            # General Topics
            'nlp': "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language for tasks like translation and sentiment analysis.",
            'chatbot': "A chatbot is an AI-powered program designed to simulate human conversation through text or voice, used for customer service, information retrieval, and entertainment.",
            'data science': "Data science combines statistics, programming, and domain expertise to extract insights from data using techniques like machine learning and data visualization."
        }
        
        # Common Q&A patterns
        self.qa_patterns = {
            ('what', 'color', 'sky'): "The sky appears blue due to a phenomenon called Rayleigh scattering, where shorter blue wavelengths are scattered more by air molecules.",
            ('what', 'capital', 'washington'): "The capital of Washington state is Olympia. If you meant Washington D.C., that's the capital of the United States.",
            ('who', 'president'): "I don't have access to current political information, but I can discuss NLP and programming topics!",
            ('what', 'time'): "I don't have access to real-time information, but I can help with NLP and programming questions!",
            ('how', 'weather'): "I don't have weather data, but I can discuss how NLP could be used to analyze weather reports!"
        }
        
        print("✅ Simple NLP Chatbot ready!")
    
    def _import_libraries(self):
        """Import all required NLP libraries"""
        try:
            # NLTK
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.nltk = nltk
            self.tokenizer = word_tokenize
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
            print("✅ NLTK loaded")
        except Exception as e:
            print(f"⚠️ NLTK issue: {e}")
            self.tokenizer = lambda x: x.split()
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but'}
        
        try:
            # TextBlob
            from textblob import TextBlob
            self.TextBlob = TextBlob
            print("✅ TextBlob loaded")
        except Exception as e:
            print(f"⚠️ TextBlob issue: {e}")
            self.TextBlob = None
        
        try:
            # spaCy
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy loaded")
        except Exception as e:
            print(f"⚠️ spaCy issue: {e}")
            self.nlp = None
        
        try:
            # ChatterBot
            from chatterbot import ChatBot
            from chatterbot.trainers import ListTrainer
            
            # Create simple chatbot
            self.chatterbot = ChatBot(
                'SimpleBot',
                storage_adapter='chatterbot.storage.SQLStorageAdapter',
                database_uri='sqlite:///simple_chatbot.db',
                logic_adapters=[
                    'chatterbot.logic.BestMatch',
                ]
            )
            
            # Train with basic conversations
            trainer = ListTrainer(self.chatterbot)
            trainer.train([
                "Hello", "Hi there! How can I help you?",
                "How are you?", "I'm doing well, thank you!",
                "What's your name?", "I'm a simple NLP chatbot!",
                "Goodbye", "Goodbye! Have a great day!",
                "Thank you", "You're welcome!",
                "Help", "I can chat with you and answer questions about NLP topics!"
            ])
            
            print("✅ ChatterBot loaded and trained")
        except Exception as e:
            print(f"⚠️ ChatterBot issue: {e}")
            self.chatterbot = None
    
    def _setup_components(self):
        """Setup chatbot components"""
        self.conversation_count = 0
        self.start_time = datetime.now()
    
    def analyze_with_textblob(self, text):
        """Analyze text with TextBlob"""
        if not self.TextBlob:
            return {"sentiment": "neutral", "polarity": 0.0}
        
        try:
            blob = self.TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "polarity": polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
        except:
            return {"sentiment": "neutral", "polarity": 0.0}
    
    def analyze_with_nltk(self, text):
        """Analyze text with NLTK"""
        try:
            # Tokenization
            tokens = self.tokenizer(text.lower())
            
            # Remove stopwords
            meaningful_words = [word for word in tokens if word.isalpha() and word not in self.stop_words]
            
            # Sentiment analysis
            try:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                compound = sentiment_scores['compound']
                
                if compound >= 0.05:
                    nltk_sentiment = "positive"
                elif compound <= -0.05:
                    nltk_sentiment = "negative"
                else:
                    nltk_sentiment = "neutral"
            except:
                nltk_sentiment = "neutral"
                sentiment_scores = {}
            
            return {
                "tokens": tokens,
                "meaningful_words": meaningful_words,
                "sentiment": nltk_sentiment,
                "sentiment_scores": sentiment_scores
            }
        except Exception as e:
            return {
                "tokens": text.split(),
                "meaningful_words": [text],
                "sentiment": "neutral"
            }
    
    def analyze_with_spacy(self, text):
        """Analyze text with spaCy"""
        if not self.nlp:
            return {"entities": [], "pos_tags": []}
        
        try:
            doc = self.nlp(text)
            
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            pos_tags = [(token.text, token.pos_) for token in doc if not token.is_space]
            
            return {
                "entities": entities,
                "pos_tags": pos_tags
            }
        except:
            return {"entities": [], "pos_tags": []}
    
    def search_knowledge(self, text):
        """Search knowledge base and Q&A patterns"""
        text_lower = text.lower()
        tokens = text_lower.split()
        
        # Check Q&A patterns first
        for pattern_words, answer in self.qa_patterns.items():
            if all(word in tokens for word in pattern_words):
                return answer
        
        # Direct keyword search in knowledge base
        for topic, answer in self.knowledge.items():
            if topic in text_lower:
                return answer
        
        # Check for question words about topics
        for topic, answer in self.knowledge.items():
            topic_words = topic.split()
            if any(word in text_lower for word in topic_words):
                return answer
        
        return None
    
    def get_chatterbot_response(self, text):
        """Get response from ChatterBot with better filtering"""
        if not self.chatterbot:
            return None
        
        try:
            response = self.chatterbot.get_response(text)
            response_text = str(response).strip()
            
            # Filter out inappropriate responses
            bad_responses = ['thank you', 'you\'re welcome', 'i\'m a simple nlp chatbot']
            text_lower = text.lower()
            
            # Only use ChatterBot for greetings and basic conversation
            if any(word in text_lower for word in ['hello', 'hi', 'hey', 'how are you', 'goodbye', 'bye']):
                if response_text and response_text.lower() not in bad_responses:
                    return response_text
        except:
            pass
        
        return None
    
    def generate_response(self, user_input):
        """Generate response using all NLP libraries"""
        if not user_input.strip():
            return "Please say something!"
        
        # Analyze with all libraries
        textblob_analysis = self.analyze_with_textblob(user_input)
        nltk_analysis = self.analyze_with_nltk(user_input)
        spacy_analysis = self.analyze_with_spacy(user_input)
        
        # Check knowledge base first
        knowledge_response = self.search_knowledge(user_input)
        if knowledge_response:
            return f"{knowledge_response}\n\n[Sentiment: {textblob_analysis['sentiment']}]"
        
        # Try ChatterBot
        chatterbot_response = self.get_chatterbot_response(user_input)
        if chatterbot_response:
            return f"{chatterbot_response}\n\n[Sentiment: {textblob_analysis['sentiment']}]"
        
        # Handle greetings
        if any(word in user_input.lower() for word in ['hello', 'hi', 'hey', 'greetings']):
            return f"Hello! I'm an NLP chatbot using NLTK, ChatterBot, TextBlob, and spaCy. How can I help you?\n\n[Sentiment: {textblob_analysis['sentiment']}]"
        
        # Handle farewells
        if any(word in user_input.lower() for word in ['bye', 'goodbye', 'farewell', 'see you']):
            return f"Goodbye! Thanks for chatting with me!\n\n[Sentiment: {textblob_analysis['sentiment']}]"
        
        # Handle questions intelligently
        if any(word in user_input.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
            entities = spacy_analysis.get('entities', [])
            meaningful_words = nltk_analysis.get('meaningful_words', [])
            
            if entities:
                entity_text = ", ".join([ent[0] for ent in entities[:2]])
                return f"You're asking about {entity_text}. While I don't have specific information on this topic, I specialize in NLP, programming, and AI. Try asking about machine learning, Python, or chatbot development!\n\n[Sentiment: {textblob_analysis['sentiment']}]"
            elif meaningful_words:
                topic_text = ', '.join(meaningful_words[:3])
                return f"That's a great question about {topic_text}! I specialize in NLP and programming topics. Try asking me about machine learning, Python, artificial intelligence, or the NLP libraries I use (NLTK, spaCy, TextBlob, ChatterBot).\n\n[Sentiment: {textblob_analysis['sentiment']}]"
            else:
                return f"I'd love to help answer your question! I specialize in NLP, programming, and AI topics. Try asking about machine learning, Python, or how chatbots work.\n\n[Sentiment: {textblob_analysis['sentiment']}]"
        
        # Handle statements or general input
        meaningful_words = nltk_analysis.get('meaningful_words', [])
        if meaningful_words:
            return f"I see you mentioned {', '.join(meaningful_words[:3])}. That's interesting! I'd be happy to discuss related NLP, programming, or AI topics with you.\n\n[Sentiment: {textblob_analysis['sentiment']}]"
        
        return f"Thanks for chatting with me! I'm here to discuss NLP, programming, AI, and machine learning topics. What would you like to know about?\n\n[Sentiment: {textblob_analysis['sentiment']}]"
    
    def detailed_analysis(self, text):
        """Get detailed NLP analysis"""
        textblob_analysis = self.analyze_with_textblob(text)
        nltk_analysis = self.analyze_with_nltk(text)
        spacy_analysis = self.analyze_with_spacy(text)
        
        report = f"""
🔍 NLP ANALYSIS REPORT
{'='*40}
📝 Text: "{text}"

🎭 SENTIMENT ANALYSIS:
   TextBlob: {textblob_analysis['sentiment']} (polarity: {textblob_analysis.get('polarity', 0):.2f})
   NLTK: {nltk_analysis['sentiment']}

🏷️ NAMED ENTITIES (spaCy):
   {', '.join([f"{ent[0]} ({ent[1]})" for ent in spacy_analysis['entities']]) if spacy_analysis['entities'] else 'None detected'}

🔤 TOKENS (NLTK):
   {', '.join(nltk_analysis['tokens'][:10])}

💭 KEY WORDS:
   {', '.join(nltk_analysis['meaningful_words'][:5])}

🎯 LIBRARIES USED:
   ✅ NLTK - Tokenization & Sentiment
   ✅ TextBlob - Sentiment Analysis  
   ✅ spaCy - Entity Recognition
   ✅ ChatterBot - Conversation
"""
        return report

def main():
    """Main chatbot interface"""
    print("=" * 60)
    print("🚀 SIMPLE NLP CHATBOT")
    print("Using: NLTK • ChatterBot • TextBlob • spaCy")
    print("=" * 60)
    
    try:
        # Create chatbot
        bot = SimpleNLPChatbot()
        
        print("\n💡 Commands:")
        print("   • Chat normally for conversation")
        print("   • Type 'analyze: [text]' for detailed NLP analysis")
        print("   • Type 'help' for available topics")
        print("   • Type 'quit' to exit")
        print("\n🎉 Start chatting!\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    response = bot.generate_response(user_input)
                    print(f"🤖 Bot: {response}")
                    break
                
                if user_input.lower().startswith('analyze:'):
                    analyze_text = user_input[8:].strip()
                    if analyze_text:
                        print(bot.detailed_analysis(analyze_text))
                    else:
                        print("🤖 Bot: Please provide text to analyze. Example: analyze: I love Python!")
                    continue
                
                if user_input.lower() == 'help':
                    help_text = """
🆘 HELP - Available Topics:
• Machine Learning
• Python Programming  
• Artificial Intelligence
• NLTK (Natural Language Toolkit)
• spaCy (Industrial NLP)
• TextBlob (Text Processing)
• ChatterBot (Conversational AI)

Try asking: "What is machine learning?" or "Tell me about Python"
"""
                    print(help_text)
                    continue
                
                # Get bot response
                response = bot.generate_response(user_input)
                print(f"🤖 Bot: {response}")
                
                bot.conversation_count += 1
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")
    
    except Exception as e:
        print(f"❌ Failed to start chatbot: {e}")
        print("\n🔧 Try running: pip3 install nltk textblob spacy chatterbot")

if __name__ == "__main__":
    main()

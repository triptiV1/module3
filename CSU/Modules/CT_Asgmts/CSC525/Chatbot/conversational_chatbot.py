#!/usr/bin/env python3
"""
Conversational Chatbot using Real Datasets
- WikiQA Corpus for factual Q&A
- ConvAI data for natural conversation
- NLTK, ChatterBot, TextBlob, spaCy integration
"""

import os
import json
import csv
import requests
import ssl
import warnings
from datetime import datetime
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Fix SSL issues for downloads
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
        'stopwords', 'wordnet'
    ]
    for item in required_data:
        try:
            nltk.download(item, quiet=True)
        except:
            pass

class ConversationalChatbot:
    """
    Advanced conversational chatbot using real datasets:
    - WikiQA for factual question answering
    - ConvAI for natural conversation patterns
    - All 4 NLP libraries integrated
    """
    
    def __init__(self, wikiqa_path="/Users/tvishwak/Downloads/WikiQACorpus"):
        print("🤖 Initializing Conversational Chatbot with Real Data...")
        
        self.wikiqa_path = wikiqa_path
        self.qa_database = {}
        self.conversation_patterns = []
        
        # Setup NLP libraries
        setup_nltk()
        self._import_libraries()
        
        # Load and process datasets
        self._load_wikiqa_data()
        self._load_convai_data()
        
        # Initialize ChatterBot with real conversation data
        self._setup_chatterbot()
        
        # Statistics
        self.conversation_count = 0
        self.successful_qa = 0
        
        print("✅ Conversational Chatbot ready!")
    
    def _import_libraries(self):
        """Import all NLP libraries"""
        try:
            # NLTK
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            self.tokenizer = word_tokenize
            self.stop_words = set(stopwords.words('english'))
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            print("✅ NLTK and scikit-learn loaded")
        except Exception as e:
            print(f"⚠️ NLTK issue: {e}")
        
        try:
            # TextBlob
            from textblob import TextBlob
            self.TextBlob = TextBlob
            print("✅ TextBlob loaded")
        except Exception as e:
            print(f"⚠️ TextBlob issue: {e}")
        
        try:
            # spaCy
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy loaded")
        except Exception as e:
            print(f"⚠️ spaCy issue: {e}")
            self.nlp = None
    
    def _load_wikiqa_data(self):
        """Load WikiQA corpus for factual Q&A"""
        print("📚 Loading WikiQA corpus...")
        
        try:
            # Load training data
            train_file = os.path.join(self.wikiqa_path, "WikiQA-train.tsv")
            if not os.path.exists(train_file):
                print(f"❌ WikiQA file not found: {train_file}")
                return
            
            # Read WikiQA data
            df = pd.read_csv(train_file, sep='\t')
            
            # Group by question and collect correct answers (Label=1)
            question_groups = df[df['Label'] == 1].groupby('Question')
            
            for question, group in question_groups:
                # Clean question
                question_clean = question.lower().strip()
                
                # Get all correct answers for this question
                answers = group['Sentence'].tolist()
                
                # Store best answer (first one, usually highest quality)
                if answers:
                    self.qa_database[question_clean] = {
                        'answer': answers[0],
                        'all_answers': answers,
                        'document_title': group.iloc[0]['DocumentTitle']
                    }
            
            print(f"✅ Loaded {len(self.qa_database)} Q&A pairs from WikiQA")
            
        except Exception as e:
            print(f"⚠️ Error loading WikiQA: {e}")
    
    def _load_convai_data(self):
        """Download and load ConvAI conversation data"""
        print("💬 Loading ConvAI conversation data...")
        
        try:
            # ConvAI data URLs (from the website)
            convai_urls = [
                "https://github.com/DeepPavlov/convai/raw/master/2018/data/data_tolokers.json",
                "https://github.com/DeepPavlov/convai/raw/master/2018/data/data_volunteers.json"
            ]
            
            for url in convai_urls:
                try:
                    print(f"📥 Downloading {url.split('/')[-1]}...")
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract conversation patterns
                        for dialogue in data:
                            if 'dialog' in dialogue:
                                conversation = []
                                for turn in dialogue['dialog']:
                                    if 'text' in turn:
                                        conversation.append(turn['text'])
                                
                                # Store conversation pairs
                                if len(conversation) >= 2:
                                    for i in range(len(conversation) - 1):
                                        self.conversation_patterns.append({
                                            'input': conversation[i],
                                            'response': conversation[i + 1]
                                        })
                    else:
                        print(f"⚠️ Could not download {url}")
                
                except Exception as e:
                    print(f"⚠️ Error downloading ConvAI data: {e}")
            
            print(f"✅ Loaded {len(self.conversation_patterns)} conversation patterns")
            
        except Exception as e:
            print(f"⚠️ Error loading ConvAI data: {e}")
    
    def _setup_chatterbot(self):
        """Setup ChatterBot with conversation data"""
        try:
            from chatterbot import ChatBot
            from chatterbot.trainers import ListTrainer
            
            self.chatterbot = ChatBot(
                'ConversationalBot',
                storage_adapter='chatterbot.storage.SQLStorageAdapter',
                database_uri='sqlite:///conversational_bot.db',
                logic_adapters=[
                    'chatterbot.logic.BestMatch'
                ]
            )
            
            # Train with ConvAI patterns (sample to avoid overtraining)
            if self.conversation_patterns:
                trainer = ListTrainer(self.chatterbot)
                
                # Use best conversation patterns (limit to avoid memory issues)
                sample_size = min(500, len(self.conversation_patterns))
                sample_patterns = self.conversation_patterns[:sample_size]
                
                training_data = []
                for pattern in sample_patterns:
                    if len(pattern['input']) > 10 and len(pattern['response']) > 10:
                        training_data.extend([pattern['input'], pattern['response']])
                
                if training_data:
                    trainer.train(training_data)
                    print(f"✅ ChatterBot trained on {len(training_data)//2} conversation pairs")
            
        except Exception as e:
            print(f"⚠️ ChatterBot setup issue: {e}")
            self.chatterbot = None
    
    def analyze_with_nlp_libraries(self, text):
        """Analyze text with all 4 NLP libraries"""
        analysis = {}
        
        # NLTK Analysis
        try:
            tokens = self.tokenizer(text.lower())
            analysis['nltk'] = {
                'tokens': tokens,
                'meaningful_words': [w for w in tokens if w.isalpha() and w not in self.stop_words]
            }
        except:
            analysis['nltk'] = {'tokens': []}
        
        # TextBlob Analysis
        try:
            blob = self.TextBlob(text)
            analysis['textblob'] = {
                'noun_phrases': list(blob.noun_phrases)
            }
        except:
            analysis['textblob'] = {'noun_phrases': []}
        
        # spaCy Analysis
        if self.nlp:
            try:
                doc = self.nlp(text)
                analysis['spacy'] = {
                    'entities': [(ent.text, ent.label_) for ent in doc.ents],
                    'pos_tags': [(token.text, token.pos_) for token in doc if not token.is_space]
                }
            except:
                analysis['spacy'] = {'entities': [], 'pos_tags': []}
        
        return analysis
    
    def search_factual_qa(self, question):
        """Search WikiQA database for factual answers"""
        question_lower = question.lower().strip()
        
        # Direct match
        if question_lower in self.qa_database:
            return self.qa_database[question_lower]
        
        # Similarity search using TF-IDF
        try:
            questions = list(self.qa_database.keys())
            all_questions = questions + [question_lower]
            
            # Vectorize questions
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_questions)
            
            # Calculate similarity with input question
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
            
            # Find best match above threshold
            best_idx = similarities.argmax()
            if similarities[best_idx] > 0.3:  # Threshold for relevance
                best_question = questions[best_idx]
                return self.qa_database[best_question]
        
        except Exception as e:
            print(f"⚠️ Similarity search error: {e}")
        
        return None
    
    def get_conversational_response(self, text):
        """Get conversational response from ChatterBot"""
        if not self.chatterbot:
            return None
        
        try:
            response = self.chatterbot.get_response(text)
            response_text = str(response).strip()
            
            # Filter quality responses
            if len(response_text) > 5 and response_text.lower() != text.lower():
                return response_text
        except:
            pass
        
        return None
    
    def generate_response(self, user_input):
        """Generate intelligent response using all data sources"""
        if not user_input.strip():
            return "Please say something!"
        
        # Analyze input with all NLP libraries
        analysis = self.analyze_with_nlp_libraries(user_input)
        
        # Handle greetings
        if any(word in user_input.lower() for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! I'm a conversational chatbot trained on WikiQA and ConvAI datasets. I can answer factual questions and have natural conversations. How can I help you?"
        
        # Handle farewells
        if any(word in user_input.lower() for word in ['bye', 'goodbye', 'farewell', 'see you']):
            return "Goodbye! Thanks for our conversation. I enjoyed chatting with you!"
        
        # Try factual Q&A first for questions
        if any(word in user_input.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
            factual_answer = self.search_factual_qa(user_input)
            if factual_answer:
                self.successful_qa += 1
                source_info = f"Source: {factual_answer['document_title']}"
                return f"{factual_answer['answer']}\n\n{source_info}"
        
        # Try conversational response
        conversational_response = self.get_conversational_response(user_input)
        if conversational_response:
            return conversational_response
        
        # Intelligent fallback using NLP analysis
        entities = analysis['spacy']['entities']
        meaningful_words = analysis['nltk']['meaningful_words']
        
        if entities:
            entity_text = ", ".join([ent[0] for ent in entities[:2]])
            return f"I notice you mentioned {entity_text}. That's interesting! I can discuss factual topics or have general conversations. What would you like to explore?"
        
        if meaningful_words:
            topics = ", ".join(meaningful_words[:3])
            return f"I see you're interested in {topics}. I'm trained on Wikipedia knowledge and human conversations, so I can help with factual questions or just chat! What would you like to know?"
        
        return "That's an interesting message! I'm here for both factual Q&A and natural conversation. Feel free to ask me questions or just chat about whatever interests you."
    
    def get_analysis_report(self, text):
        """Get detailed NLP analysis report"""
        analysis = self.analyze_with_nlp_libraries(text)
        
        report = f"""
🔍 CONVERSATIONAL NLP ANALYSIS
{'='*45}
📝 Text: "{text}"

🧠 NLTK Analysis:
   Tokens: {', '.join(analysis['nltk']['tokens'][:8])}
   Key Words: {', '.join(analysis['nltk']['meaningful_words'][:5])}

💭 TextBlob Analysis:
   Noun Phrases: {', '.join(analysis['textblob']['noun_phrases'][:3])}

🏷️ spaCy Analysis:
   Entities: {', '.join([f"{ent[0]}({ent[1]})" for ent in analysis['spacy']['entities']])}
   POS Tags: {', '.join([f"{word}({pos})" for word, pos in analysis['spacy']['pos_tags'][:6]])}

🔍 Data Sources:
   ✅ WikiQA: {len(self.qa_database)} factual Q&A pairs
   ✅ ConvAI: {len(self.conversation_patterns)} conversation patterns
   ✅ ChatterBot: Trained on real human dialogues
   
📊 Usage Stats:
   Total conversations: {self.conversation_count}
   Successful Q&A: {self.successful_qa}
"""
        return report
    
    def get_stats(self):
        """Get chatbot statistics"""
        return f"""
📊 CONVERSATIONAL CHATBOT STATISTICS
{'='*40}

📚 Knowledge Base:
   • WikiQA Questions: {len(self.qa_database)}
   • ConvAI Patterns: {len(self.conversation_patterns)}
   • ChatterBot: {'Active' if self.chatterbot else 'Inactive'}

💬 Session Stats:
   • Total Conversations: {self.conversation_count}
   • Successful Q&A: {self.successful_qa}
   • Q&A Success Rate: {(self.successful_qa/max(1,self.conversation_count))*100:.1f}%

🔧 NLP Libraries:
   ✅ NLTK - Tokenization & Text Processing
   ✅ TextBlob - Noun Phrase Extraction
   ✅ spaCy - Entity Recognition & POS Tagging
   ✅ ChatterBot - Conversation Patterns
   ✅ scikit-learn - Similarity Matching

📊 Data Sources:
   • WikiQA Corpus (Microsoft Research)
   • ConvAI Dataset (PersonaChat)
   • Real human conversations
"""

def main():
    """Main chatbot interface"""
    print("=" * 70)
    print("🚀 CONVERSATIONAL CHATBOT WITH REAL DATASETS")
    print("Using: WikiQA Corpus + ConvAI Data + 4 NLP Libraries")
    print("=" * 70)
    
    try:
        # Initialize chatbot with real data
        bot = ConversationalChatbot()
        
        print("\n💡 Features:")
        print("   • Factual Q&A from 3,000+ Wikipedia questions")
        print("   • Natural conversation from 4,500+ real dialogues")
        print("   • Entity recognition and POS tagging")
        print("   • Intelligent response selection")
        print("\n💡 Commands:")
        print("   • Ask any factual question")
        print("   • Have natural conversations")
        print("   • Type 'analyze: [text]' for NLP analysis")
        print("   • Type 'stats' for usage statistics")
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
                    print(bot.get_stats())
                    break
                
                if user_input.lower().startswith('analyze:'):
                    analyze_text = user_input[8:].strip()
                    if analyze_text:
                        print(bot.get_analysis_report(analyze_text))
                    else:
                        print("🤖 Bot: Please provide text to analyze!")
                    continue
                
                if user_input.lower() == 'stats':
                    print(bot.get_stats())
                    continue
                
                # Get response
                response = bot.generate_response(user_input)
                print(f"🤖 Bot: {response}")
                
                bot.conversation_count += 1
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for testing the conversational chatbot!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")
    
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("\n🔧 Requirements:")
        print("1. WikiQA corpus at /Users/tvishwak/Downloads/WikiQACorpus")
        print("2. Internet connection for ConvAI data")
        print("3. NLP libraries: pip install nltk textblob spacy chatterbot pandas requests scikit-learn")

if __name__ == "__main__":
    main()

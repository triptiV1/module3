#!/usr/bin/env python3
"""
Standalone NLP Chatbot Training System
CSC525 - Machine Learning
Author: Tripti Vishwakarma

A comprehensive chatbot training system with intent classification,
response generation, and multiple neural network architectures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Dropout, Bidirectional, 
    GlobalMaxPooling1D, Conv1D, MaxPooling1D, Input,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ChatbotTrainer:
    """
    Standalone Chatbot Training System
    
    Features:
    - Intent classification using neural networks
    - Response generation and mapping
    - Multiple model architectures (LSTM, CNN-LSTM, Transformer)
    - Interactive chatbot demo
    - Comprehensive evaluation and visualization
    """
    
    def __init__(self, dataset_path='chatbot_training_dataset.csv'):
        """
        Initialize the chatbot trainer
        
        Args:
            dataset_path (str): Path to the chatbot training dataset
        """
        self.dataset_path = dataset_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.intent_responses = {}
        
        # Model parameters
        self.max_sequence_length = 50
        self.vocab_size = 5000
        self.embedding_dim = 100
        
        # Training hyperparameters
        self.hyperparameters = {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'lstm_units': 128,
            'dense_units': 64,
            'attention_heads': 8,
            'transformer_layers': 2
        }
        
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        print("Chatbot Training System Initialized")
        print(f"Dataset: {dataset_path}")
        print(f"Hyperparameters: {self.hyperparameters}")
    
    def load_dataset(self):
        """Load and analyze the chatbot training dataset"""
        print(f"\n=== Loading Chatbot Dataset ===")
        
        # Load data
        self.df = pd.read_csv(self.dataset_path)
        print(f"Loaded {len(self.df)} training examples")
        
        # Create intent-response mapping
        for _, row in self.df.iterrows():
            intent = row['intent']
            response = row['response']
            if intent not in self.intent_responses:
                self.intent_responses[intent] = []
            if response not in self.intent_responses[intent]:
                self.intent_responses[intent].append(response)
        
        # Display dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total intents: {len(self.df['intent'].unique())}")
        print(f"Total examples: {len(self.df)}")
        
        intent_counts = self.df['intent'].value_counts()
        print(f"\nIntent Distribution:")
        for intent, count in intent_counts.items():
            print(f"  {intent}: {count} examples")
        
        # Visualize intent distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        intent_counts.plot(kind='bar', color='skyblue')
        plt.title('Intent Distribution')
        plt.xlabel('Intent')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%')
        plt.title('Intent Distribution (Pie Chart)')
        
        # Text length analysis
        text_lengths = self.df['text'].str.len()
        plt.subplot(2, 2, 3)
        plt.hist(text_lengths, bins=20, color='lightgreen', alpha=0.7)
        plt.title('Text Length Distribution')
        plt.xlabel('Character Count')
        plt.ylabel('Frequency')
        
        # Word count analysis
        word_counts = self.df['text'].str.split().str.len()
        plt.subplot(2, 2, 4)
        plt.hist(word_counts, bins=15, color='orange', alpha=0.7)
        plt.title('Word Count Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('chatbot_dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df
    
    def preprocess_text(self, text):
        """
        Preprocess text for training
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters except basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\?\!\.]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize (keep important words for chatbot)
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if len(token) > 1  # Keep most words for chatbot context
        ]
        
        return ' '.join(tokens)
    
    def prepare_training_data(self):
        """Prepare data for neural network training"""
        print(f"\n=== Preparing Training Data ===")
        
        # Preprocess text
        self.df['processed_text'] = self.df['text'].apply(self.preprocess_text)
        
        # Encode intents
        self.label_encoder = LabelEncoder()
        self.df['encoded_intent'] = self.label_encoder.fit_transform(self.df['intent'])
        
        print(f"Intent mapping:")
        for i, intent in enumerate(self.label_encoder.classes_):
            print(f"  {intent}: {i}")
        
        # Tokenize text
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.df['processed_text'])
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(self.df['processed_text'])
        self.X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        self.y = tf.keras.utils.to_categorical(self.df['encoded_intent'])
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Input shape: {self.X.shape}")
        print(f"Output shape: {self.y.shape}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_lstm_model(self):
        """Build LSTM-based intent classification model"""
        print(f"\n=== Building LSTM Model ===")
        
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                name='embedding'
            ),
            Bidirectional(LSTM(
                self.hyperparameters['lstm_units'],
                dropout=self.hyperparameters['dropout_rate'],
                recurrent_dropout=self.hyperparameters['dropout_rate'],
                return_sequences=True,
                name='lstm_1'
            )),
            Bidirectional(LSTM(
                self.hyperparameters['lstm_units'] // 2,
                dropout=self.hyperparameters['dropout_rate'],
                recurrent_dropout=self.hyperparameters['dropout_rate'],
                name='lstm_2'
            )),
            Dense(self.hyperparameters['dense_units'], activation='relu', name='dense_1'),
            Dropout(self.hyperparameters['dropout_rate'], name='dropout'),
            Dense(len(self.label_encoder.classes_), activation='softmax', name='output')
        ])
        
        return model
    
    def build_cnn_lstm_model(self):
        """Build CNN-LSTM hybrid model"""
        print(f"\n=== Building CNN-LSTM Model ===")
        
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                name='embedding'
            ),
            Conv1D(filters=128, kernel_size=3, activation='relu', name='conv1d_1'),
            MaxPooling1D(pool_size=2, name='maxpool_1'),
            Conv1D(filters=64, kernel_size=3, activation='relu', name='conv1d_2'),
            LSTM(
                self.hyperparameters['lstm_units'],
                dropout=self.hyperparameters['dropout_rate'],
                recurrent_dropout=self.hyperparameters['dropout_rate'],
                name='lstm'
            ),
            Dense(self.hyperparameters['dense_units'], activation='relu', name='dense'),
            Dropout(self.hyperparameters['dropout_rate'], name='dropout'),
            Dense(len(self.label_encoder.classes_), activation='softmax', name='output')
        ])
        
        return model
    
    def build_transformer_model(self):
        """Build Transformer-based model"""
        print(f"\n=== Building Transformer Model ===")
        
        # Input layer
        inputs = Input(shape=(self.max_sequence_length,), name='input')
        
        # Embedding
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name='embedding'
        )(inputs)
        
        # Transformer blocks
        x = embedding
        for i in range(self.hyperparameters['transformer_layers']):
            # Multi-head attention
            attention = MultiHeadAttention(
                num_heads=self.hyperparameters['attention_heads'],
                key_dim=self.embedding_dim // self.hyperparameters['attention_heads'],
                name=f'attention_{i}'
            )(x, x)
            
            # Add & Norm
            x = LayerNormalization(name=f'norm_1_{i}')(x + attention)
            
            # Feed forward
            ff = Dense(self.embedding_dim * 2, activation='relu', name=f'ff_1_{i}')(x)
            ff = Dense(self.embedding_dim, name=f'ff_2_{i}')(ff)
            
            # Add & Norm
            x = LayerNormalization(name=f'norm_2_{i}')(x + ff)
        
        # Global pooling and classification
        x = GlobalAveragePooling1D(name='global_avg_pool')(x)
        x = Dense(self.hyperparameters['dense_units'], activation='relu', name='dense')(x)
        x = Dropout(self.hyperparameters['dropout_rate'], name='dropout')(x)
        outputs = Dense(len(self.label_encoder.classes_), activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='transformer_chatbot')
        return model
    
    def train_model(self, model_type='lstm'):
        """
        Train the chatbot model
        
        Args:
            model_type (str): Type of model ('lstm', 'cnn_lstm', 'transformer')
        """
        print(f"\n=== Training {model_type.upper()} Chatbot Model ===")
        
        # Build model
        if model_type == 'lstm':
            self.model = self.build_lstm_model()
        elif model_type == 'cnn_lstm':
            self.model = self.build_cnn_lstm_model()
        elif model_type == 'transformer':
            self.model = self.build_transformer_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.hyperparameters['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n{model_type.upper()} Model Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_chatbot_{model_type}_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            batch_size=self.hyperparameters['batch_size'],
            epochs=self.hyperparameters['epochs'],
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, model_type='lstm'):
        """Evaluate the trained model"""
        print(f"\n=== Evaluating {model_type.upper()} Model ===")
        
        # Predictions
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Visualization
        plt.figure(figsize=(15, 12))
        
        # Training history
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title(f'{model_type.upper()} Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title(f'{model_type.upper()} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion matrix
        plt.subplot(2, 3, 3)
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Intent')
        plt.ylabel('True Intent')
        
        # Intent accuracy per class
        plt.subplot(2, 3, 4)
        class_accuracy = []
        for i, intent in enumerate(self.label_encoder.classes_):
            mask = y_true == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == y_true[mask]).mean()
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        plt.bar(range(len(class_accuracy)), class_accuracy, color='lightcoral')
        plt.title('Accuracy per Intent')
        plt.xlabel('Intent Index')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(self.label_encoder.classes_)), 
                  [f"{i}" for i in range(len(self.label_encoder.classes_))], rotation=45)
        
        # Model parameters
        plt.subplot(2, 3, 5)
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        plt.bar(['Total', 'Trainable'], [total_params, trainable_params], color=['skyblue', 'orange'])
        plt.title('Model Parameters')
        plt.ylabel('Parameter Count')
        
        # Training metrics summary
        plt.subplot(2, 3, 6)
        final_metrics = {
            'Final Accuracy': self.history.history['val_accuracy'][-1],
            'Best Accuracy': max(self.history.history['val_accuracy']),
            'Final Loss': self.history.history['val_loss'][-1],
            'Best Loss': min(self.history.history['val_loss'])
        }
        
        metrics_names = list(final_metrics.keys())
        metrics_values = list(final_metrics.values())
        
        plt.bar(range(len(metrics_names)), metrics_values, color='lightgreen')
        plt.title('Training Metrics Summary')
        plt.xticks(range(len(metrics_names)), metrics_names, rotation=45)
        plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(f'chatbot_{model_type}_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_pred, y_true
    
    def predict_intent(self, text):
        """
        Predict intent for input text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Predict
        prediction_proba = self.model.predict(padded_sequence, verbose=0)[0]
        predicted_intent_idx = np.argmax(prediction_proba)
        predicted_intent = self.label_encoder.inverse_transform([predicted_intent_idx])[0]
        confidence = prediction_proba[predicted_intent_idx]
        
        return {
            'text': text,
            'predicted_intent': predicted_intent,
            'confidence': float(confidence),
            'probabilities': {
                intent: float(prob) 
                for intent, prob in zip(self.label_encoder.classes_, prediction_proba)
            }
        }
    
    def generate_response(self, intent):
        """
        Generate response for predicted intent
        
        Args:
            intent (str): Predicted intent
            
        Returns:
            str: Generated response
        """
        if intent in self.intent_responses:
            responses = self.intent_responses[intent]
            return np.random.choice(responses)
        else:
            return "I'm not sure how to respond to that. Can you try rephrasing?"
    
    def chat_with_bot(self):
        """Interactive chatbot demo"""
        print(f"\n=== Interactive Chatbot Demo ===")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("Bot: Goodbye! Have a great day!")
                break
            
            if not user_input:
                print("Bot: Please say something!")
                continue
            
            try:
                # Predict intent
                prediction = self.predict_intent(user_input)
                intent = prediction['predicted_intent']
                confidence = prediction['confidence']
                
                # Generate response
                response = self.generate_response(intent)
                
                print(f"Bot: {response}")
                print(f"[Intent: {intent}, Confidence: {confidence:.3f}]")
                
            except Exception as e:
                print(f"Bot: Sorry, I encountered an error: {e}")
    
    def save_model(self, model_type='lstm'):
        """Save trained model and artifacts"""
        print(f"\n=== Saving {model_type.upper()} Model ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"chatbot_{model_type}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        self.model.save(f"{model_dir}/chatbot_model.h5")
        
        # Save tokenizer
        with open(f"{model_dir}/tokenizer.pkl", 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save label encoder
        with open(f"{model_dir}/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save intent responses
        with open(f"{model_dir}/intent_responses.json", 'w') as f:
            json.dump(self.intent_responses, f, indent=2)
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'hyperparameters': self.hyperparameters,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim,
            'intents': self.label_encoder.classes_.tolist(),
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'timestamp': timestamp
        }
        
        with open(f"{model_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to: {model_dir}")
        return model_dir


def main():
    """Main training pipeline"""
    print("=== Standalone NLP Chatbot Training System ===")
    print("CSC525 - Machine Learning")
    print("Author: Tripti Vishwakarma\n")
    
    # Initialize trainer
    trainer = ChatbotTrainer('chatbot_training_dataset.csv')
    
    # Load and prepare data
    trainer.load_dataset()
    trainer.prepare_training_data()
    
    # Train different model architectures
    model_types = ['lstm', 'cnn_lstm', 'transformer']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"TRAINING {model_type.upper()} CHATBOT MODEL")
        print(f"{'='*60}")
        
        try:
            # Train model
            trainer.train_model(model_type)
            
            # Evaluate model
            accuracy, y_pred, y_true = trainer.evaluate_model(model_type)
            
            # Save model
            model_dir = trainer.save_model(model_type)
            
            results[model_type] = {
                'accuracy': accuracy,
                'model_dir': model_dir
            }
            
            print(f"\n{model_type.upper()} Model Training Complete!")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Model saved to: {model_dir}")
            
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
            results[model_type] = {'error': str(e)}
    
    # Compare results
    print(f"\n{'='*60}")
    print("CHATBOT MODEL COMPARISON")
    print(f"{'='*60}")
    
    best_accuracy = 0
    best_model_type = None
    
    for model_type, result in results.items():
        if 'accuracy' in result:
            accuracy = result['accuracy']
            print(f"{model_type.upper()}: {accuracy:.4f} accuracy")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_type = model_type
        else:
            print(f"{model_type.upper()}: Failed - {result.get('error', 'Unknown error')}")
    
    if best_model_type:
        print(f"\nBest Model: {best_model_type.upper()} with {best_accuracy:.4f} accuracy")
        
        # Interactive demo with best model
        print(f"\nStarting interactive demo with {best_model_type.upper()} model...")
        trainer.chat_with_bot()
    
    print("\nChatbot training completed!")


if __name__ == "__main__":
    main()

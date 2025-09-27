#!/usr/bin/env python3
"""
NLP Chatbot Training System
CSC525 - Machine Learning
Author: Tripti Vishwakarma

This module implements a comprehensive NLP chatbot training system with multiple
model architectures including LSTM, Transformer, and BERT-based approaches.
Supports sentiment analysis and intent classification for chatbot applications.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
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

class NLPChatbotTrainer:
    """
    Comprehensive NLP Chatbot Training System
    
    Supports multiple model architectures:
    - LSTM-based sequence models
    - CNN-LSTM hybrid models
    - Transformer-based attention models
    - BERT-style pre-trained models
    """
    
    def __init__(self, data_path='augmented_text_dataset.csv', model_type='lstm'):
        """
        Initialize the chatbot trainer
        
        Args:
            data_path (str): Path to the training dataset
            model_type (str): Type of model ('lstm', 'cnn_lstm', 'transformer', 'bert')
        """
        self.data_path = data_path
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_sequence_length = 100
        self.vocab_size = 10000
        self.embedding_dim = 128
        self.history = None
        
        # Training hyperparameters
        self.hyperparameters = {
            'batch_size': 16,
            'epochs': 50,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'lstm_units': 64,
            'dense_units': 32,
            'attention_heads': 4,
            'transformer_layers': 2
        }
        
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        print(f"NLP Chatbot Trainer initialized with {model_type} architecture")
        print(f"Hyperparameters: {self.hyperparameters}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the training data"""
        print(f"\n=== Loading Data from {self.data_path} ===")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} samples")
        
        # Clean label column - remove _aug suffix for consistency
        self.df['label'] = self.df['label'].str.replace('_aug', '')
        
        # Display data distribution
        print("\nData Distribution:")
        label_counts = self.df['label'].value_counts()
        print(label_counts)
        
        # Visualize data distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        label_counts.plot(kind='bar', color=['green', 'red', 'blue'])
        plt.title('Label Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
        plt.title('Label Distribution (Pie Chart)')
        
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df
    
    def preprocess_text(self, text):
        """
        Advanced text preprocessing pipeline
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def prepare_data_for_training(self):
        """Prepare data for neural network training"""
        print("\n=== Preparing Data for Training ===")
        
        # Preprocess text data
        print("Preprocessing text data...")
        self.df['processed_text'] = self.df['text'].apply(self.preprocess_text)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['encoded_label'] = self.label_encoder.fit_transform(self.df['label'])
        
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Tokenize text
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.df['processed_text'])
        
        # Convert text to sequences
        sequences = self.tokenizer.texts_to_sequences(self.df['processed_text'])
        self.X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        self.y = tf.keras.utils.to_categorical(self.df['encoded_label'])
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Sequence shape: {self.X.shape}")
        print(f"Label shape: {self.y.shape}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_lstm_model(self):
        """Build LSTM-based model"""
        print("\n=== Building LSTM Model ===")
        
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            ),
            Bidirectional(LSTM(
                self.hyperparameters['lstm_units'],
                dropout=self.hyperparameters['dropout_rate'],
                recurrent_dropout=self.hyperparameters['dropout_rate'],
                return_sequences=True
            )),
            Bidirectional(LSTM(
                self.hyperparameters['lstm_units'] // 2,
                dropout=self.hyperparameters['dropout_rate'],
                recurrent_dropout=self.hyperparameters['dropout_rate']
            )),
            Dense(self.hyperparameters['dense_units'], activation='relu'),
            Dropout(self.hyperparameters['dropout_rate']),
            Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        return model
    
    def build_cnn_lstm_model(self):
        """Build CNN-LSTM hybrid model"""
        print("\n=== Building CNN-LSTM Hybrid Model ===")
        
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            ),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            LSTM(
                self.hyperparameters['lstm_units'],
                dropout=self.hyperparameters['dropout_rate'],
                recurrent_dropout=self.hyperparameters['dropout_rate']
            ),
            Dense(self.hyperparameters['dense_units'], activation='relu'),
            Dropout(self.hyperparameters['dropout_rate']),
            Dense(len(self.label_encoder.classes_), activation='softmax')
        ])
        
        return model
    
    def build_transformer_model(self):
        """Build Transformer-based model with multi-head attention"""
        print("\n=== Building Transformer Model ===")
        
        # Input layer
        inputs = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim
        )(inputs)
        
        # Multi-head attention layers
        x = embedding
        for _ in range(self.hyperparameters['transformer_layers']):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.hyperparameters['attention_heads'],
                key_dim=self.embedding_dim // self.hyperparameters['attention_heads']
            )(x, x)
            
            # Add & Norm
            x = LayerNormalization()(x + attention_output)
            
            # Feed forward
            ff_output = Dense(self.embedding_dim * 2, activation='relu')(x)
            ff_output = Dense(self.embedding_dim)(ff_output)
            
            # Add & Norm
            x = LayerNormalization()(x + ff_output)
        
        # Global pooling and classification
        x = GlobalAveragePooling1D()(x)
        x = Dense(self.hyperparameters['dense_units'], activation='relu')(x)
        x = Dropout(self.hyperparameters['dropout_rate'])(x)
        outputs = Dense(len(self.label_encoder.classes_), activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_model(self):
        """Build the specified model architecture"""
        if self.model_type == 'lstm':
            self.model = self.build_lstm_model()
        elif self.model_type == 'cnn_lstm':
            self.model = self.build_cnn_lstm_model()
        elif self.model_type == 'transformer':
            self.model = self.build_transformer_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.hyperparameters['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n{self.model_type.upper()} Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train_model(self):
        """Train the model with advanced callbacks"""
        print(f"\n=== Training {self.model_type.upper()} Model ===")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_{self.model_type}_chatbot_model.h5',
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
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print(f"\n=== Evaluating {self.model_type.upper()} Model ===")
        
        # Predictions
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        
        # Plot training history
        plt.subplot(2, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{self.model_type.upper()} Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title(f'{self.model_type.upper()} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Confusion matrix
        plt.subplot(2, 2, 3)
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Feature importance (for LSTM models)
        if hasattr(self.model, 'layers'):
            plt.subplot(2, 2, 4)
            layer_names = [layer.name for layer in self.model.layers if hasattr(layer, 'trainable_weights')]
            param_counts = [layer.count_params() for layer in self.model.layers if hasattr(layer, 'trainable_weights')]
            
            if param_counts:
                plt.bar(range(len(param_counts)), param_counts)
                plt.title('Parameters per Layer')
                plt.xlabel('Layer Index')
                plt.ylabel('Parameter Count')
                plt.xticks(range(len(layer_names)), [f"L{i}" for i in range(len(layer_names))], rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_type}_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_pred, y_true
    
    def save_model_and_artifacts(self):
        """Save trained model and preprocessing artifacts"""
        print(f"\n=== Saving {self.model_type.upper()} Model and Artifacts ===")
        
        # Create timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"chatbot_model_{self.model_type}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        self.model.save(f"{model_dir}/chatbot_model.h5")
        
        # Save tokenizer
        with open(f"{model_dir}/tokenizer.pkl", 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # Save label encoder
        with open(f"{model_dir}/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save hyperparameters and metadata
        metadata = {
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dim': self.embedding_dim,
            'classes': self.label_encoder.classes_.tolist(),
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'timestamp': timestamp
        }
        
        with open(f"{model_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model and artifacts saved to: {model_dir}")
        return model_dir
    
    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        
        # Predict
        prediction_proba = self.model.predict(padded_sequence)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = prediction_proba[predicted_class_idx]
        
        return {
            'text': text,
            'predicted_sentiment': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.label_encoder.classes_, prediction_proba)
            }
        }
    
    def interactive_demo(self):
        """Interactive demonstration of the trained chatbot"""
        print(f"\n=== Interactive {self.model_type.upper()} Chatbot Demo ===")
        print("Enter text to analyze sentiment (type 'quit' to exit):")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            try:
                result = self.predict_sentiment(user_input)
                print(f"\nChatbot Analysis:")
                print(f"Sentiment: {result['predicted_sentiment']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Probabilities:")
                for sentiment, prob in result['probabilities'].items():
                    print(f"  {sentiment}: {prob:.3f}")
            
            except Exception as e:
                print(f"Error analyzing text: {e}")


def main():
    """Main training pipeline"""
    print("=== NLP Chatbot Training System ===")
    print("CSC525 - Machine Learning Assignment")
    print("Author: Tripti Vishwakarma\n")
    
    # Model types to train and compare
    model_types = ['lstm', 'cnn_lstm', 'transformer']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"TRAINING {model_type.upper()} MODEL")
        print(f"{'='*60}")
        
        try:
            # Initialize trainer
            trainer = NLPChatbotTrainer(
                data_path='augmented_text_dataset.csv',
                model_type=model_type
            )
            
            # Load and preprocess data
            trainer.load_and_preprocess_data()
            trainer.prepare_data_for_training()
            
            # Build and train model
            trainer.build_model()
            trainer.train_model()
            
            # Evaluate model
            accuracy, y_pred, y_true = trainer.evaluate_model()
            
            # Save model
            model_dir = trainer.save_model_and_artifacts()
            
            # Store results
            results[model_type] = {
                'accuracy': accuracy,
                'model_dir': model_dir,
                'trainer': trainer
            }
            
            print(f"\n{model_type.upper()} Model completed successfully!")
            print(f"Final Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
            results[model_type] = {'error': str(e)}
    
    # Compare results
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    
    for model_type, result in results.items():
        if 'accuracy' in result:
            print(f"{model_type.upper()}: {result['accuracy']:.4f} accuracy")
        else:
            print(f"{model_type.upper()}: Failed - {result.get('error', 'Unknown error')}")
    
    # Interactive demo with best model
    best_model_type = max(
        [k for k, v in results.items() if 'accuracy' in v],
        key=lambda k: results[k]['accuracy'],
        default=None
    )
    
    if best_model_type:
        print(f"\nBest performing model: {best_model_type.upper()}")
        print("Starting interactive demo...")
        results[best_model_type]['trainer'].interactive_demo()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Text Data Augmentation Tool for NLP Projects
CSC525 - Machine Learning Assignment

This script implements various text augmentation techniques to expand training datasets
for natural language processing tasks. It includes synonym replacement, random insertion,
random swap, random deletion, and back-translation methods.

Author: Tripti Vishwakarma
Date: August 16, 2025
"""

import random
import re
import pandas as pd
import argparse
import os
from typing import List, Tuple
import json
import string

# Try to import NLTK, but provide fallback if not available
try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
    
    # Test if NLTK functions work properly
    try:
        # Test tokenization
        test_tokens = word_tokenize("test sentence")
        # Test wordnet
        test_syns = list(wordnet.synsets("good"))
        NLTK_AVAILABLE = True
    except:
        NLTK_AVAILABLE = False
            
except ImportError:
    NLTK_AVAILABLE = False

class TextAugmenter:
    """
    A comprehensive text augmentation class implementing multiple techniques
    for expanding NLP training datasets.
    """
    
    def __init__(self, random_seed=42):
        """Initialize the TextAugmenter with optional random seed for reproducibility."""
        random.seed(random_seed)
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                              'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'])
    
    def simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization fallback when NLTK is not available."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a given word using WordNet or fallback dictionary.
        
        Args:
            word (str): The word to find synonyms for
            
        Returns:
            List[str]: List of synonyms
        """
        if NLTK_AVAILABLE:
            try:
                synonyms = set()
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != word.lower() and synonym.isalpha():
                            synonyms.add(synonym)
                return list(synonyms)
            except:
                pass
        
        # Fallback synonym dictionary for common words
        synonym_dict = {
            'good': ['great', 'excellent', 'wonderful', 'amazing', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'disappointing'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['tiny', 'little', 'mini', 'compact', 'petite'],
            'fast': ['quick', 'rapid', 'speedy', 'swift', 'hasty'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'delayed', 'unhurried'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased', 'content'],
            'sad': ['unhappy', 'sorrowful', 'depressed', 'melancholy', 'gloomy'],
            'love': ['adore', 'cherish', 'treasure', 'appreciate', 'enjoy'],
            'hate': ['despise', 'loathe', 'detest', 'dislike', 'abhor'],
            'nice': ['pleasant', 'lovely', 'delightful', 'charming', 'agreeable'],
            'perfect': ['flawless', 'ideal', 'excellent', 'superb', 'outstanding'],
            'terrible': ['awful', 'dreadful', 'horrible', 'atrocious', 'appalling'],
            'amazing': ['incredible', 'fantastic', 'wonderful', 'remarkable', 'extraordinary']
        }
        
        return synonym_dict.get(word.lower(), [])
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n random words with their synonyms.
        
        Args:
            text (str): Input text
            n (int): Number of words to replace
            
        Returns:
            str: Augmented text with synonym replacements
        """
        words = word_tokenize(text) if NLTK_AVAILABLE else self.simple_tokenize(text)
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalpha() and word.lower() not in self.stop_words]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n synonyms into the sentence.
        
        Args:
            text (str): Input text
            n (int): Number of synonyms to insert
            
        Returns:
            str: Augmented text with random insertions
        """
        words = word_tokenize(text) if NLTK_AVAILABLE else self.simple_tokenize(text)
        for _ in range(n):
            new_word = self._add_word(words)
            if new_word:
                words.insert(random.randint(0, len(words)), new_word)
        return ' '.join(words)
    
    def _add_word(self, words: List[str]) -> str:
        """Helper method to find a synonym for random insertion."""
        random_word_list = [word for word in words if word.isalpha() and word.lower() not in self.stop_words]
        if not random_word_list:
            return None
        
        random_word = random.choice(random_word_list)
        synonyms = self.get_synonyms(random_word)
        if synonyms:
            return random.choice(synonyms)
        return None
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap the positions of two words n times.
        
        Args:
            text (str): Input text
            n (int): Number of swaps to perform
            
        Returns:
            str: Augmented text with random swaps
        """
        words = word_tokenize(text) if NLTK_AVAILABLE else self.simple_tokenize(text)
        for _ in range(n):
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words from the sentence with probability p.
        
        Args:
            text (str): Input text
            p (float): Probability of deleting each word
            
        Returns:
            str: Augmented text with random deletions
        """
        words = word_tokenize(text) if NLTK_AVAILABLE else self.simple_tokenize(text)
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)
        
        # If all words are deleted, return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return words[rand_int]
        
        return ' '.join(new_words)
    
    def back_translation_simulation(self, text: str) -> str:
        """
        Simulate back-translation by applying multiple transformations.
        This is a simplified version since actual translation requires external APIs.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Augmented text simulating back-translation effects
        """
        # Apply multiple light transformations to simulate translation artifacts
        augmented = self.synonym_replacement(text, n=2)
        augmented = self.random_swap(augmented, n=1)
        return augmented
    
    def paraphrase_generation(self, text: str) -> str:
        """
        Generate paraphrases by combining multiple augmentation techniques.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Paraphrased text
        """
        # Combine synonym replacement and random insertion for paraphrasing
        augmented = self.synonym_replacement(text, n=2)
        augmented = self.random_insertion(augmented, n=1)
        return augmented
    
    def augment_text(self, text: str, num_aug: int = 4, techniques: List[str] = None) -> List[str]:
        """
        Apply multiple augmentation techniques to generate variations of input text.
        
        Args:
            text (str): Input text to augment
            num_aug (int): Number of augmented versions to generate
            techniques (List[str]): List of techniques to use
            
        Returns:
            List[str]: List of augmented texts
        """
        if techniques is None:
            techniques = ['synonym_replacement', 'random_insertion', 'random_swap', 
                         'random_deletion', 'paraphrase_generation']
        
        augmented_texts = []
        
        for i in range(num_aug):
            technique = random.choice(techniques)
            
            if technique == 'synonym_replacement':
                aug_text = self.synonym_replacement(text, n=random.randint(1, 3))
            elif technique == 'random_insertion':
                aug_text = self.random_insertion(text, n=random.randint(1, 2))
            elif technique == 'random_swap':
                aug_text = self.random_swap(text, n=random.randint(1, 2))
            elif technique == 'random_deletion':
                aug_text = self.random_deletion(text, p=random.uniform(0.05, 0.15))
            elif technique == 'back_translation':
                aug_text = self.back_translation_simulation(text)
            elif technique == 'paraphrase_generation':
                aug_text = self.paraphrase_generation(text)
            else:
                aug_text = self.synonym_replacement(text, n=1)
            
            # Ensure the augmented text is different from original
            if aug_text != text and aug_text not in augmented_texts:
                augmented_texts.append(aug_text)
        
        return augmented_texts

def load_dataset(file_path: str) -> List[Tuple[str, str]]:
    """
    Load dataset from various file formats (CSV, TXT, JSON).
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        List[Tuple[str, str]]: List of (text, label) pairs
    """
    data = []
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # Assume first column is text, second is label (if exists)
        if len(df.columns) >= 2:
            for _, row in df.iterrows():
                data.append((str(row.iloc[0]), str(row.iloc[1])))
        else:
            for _, row in df.iterrows():
                data.append((str(row.iloc[0]), "unlabeled"))
    
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line:
                    data.append((line, f"sample_{i}"))
    
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            if isinstance(json_data, list):
                for i, item in enumerate(json_data):
                    if isinstance(item, dict):
                        text = item.get('text', item.get('sentence', str(item)))
                        label = item.get('label', item.get('category', f"sample_{i}"))
                        data.append((str(text), str(label)))
                    else:
                        data.append((str(item), f"sample_{i}"))
    
    return data

def save_augmented_dataset(original_data: List[Tuple[str, str]], 
                          augmented_data: List[Tuple[str, str]], 
                          output_path: str):
    """
    Save the augmented dataset to a file.
    
    Args:
        original_data: Original dataset
        augmented_data: Augmented dataset
        output_path: Path to save the combined dataset
    """
    all_data = original_data + augmented_data
    
    if output_path.endswith('.csv'):
        df = pd.DataFrame(all_data, columns=['text', 'label'])
        df.to_csv(output_path, index=False)
    
    elif output_path.endswith('.json'):
        json_data = [{'text': text, 'label': label, 'augmented': i >= len(original_data)} 
                    for i, (text, label) in enumerate(all_data)]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    else:  # Default to txt
        with open(output_path, 'w', encoding='utf-8') as f:
            for text, label in all_data:
                f.write(f"{text}\t{label}\n")

def main():
    """Main function to run the text augmentation script."""
    parser = argparse.ArgumentParser(description='Text Data Augmentation Tool')
    parser.add_argument('--input', '-i', required=True, help='Input dataset file path')
    parser.add_argument('--output', '-o', required=True, help='Output augmented dataset file path')
    parser.add_argument('--num_aug', '-n', type=int, default=4, 
                       help='Number of augmented versions per text (default: 4)')
    parser.add_argument('--techniques', '-t', nargs='+', 
                       choices=['synonym_replacement', 'random_insertion', 'random_swap', 
                               'random_deletion', 'back_translation', 'paraphrase_generation'],
                       help='Augmentation techniques to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return
    
    print(f"Loading dataset from: {args.input}")
    original_data = load_dataset(args.input)
    print(f"Loaded {len(original_data)} original samples")
    
    # Initialize augmenter
    augmenter = TextAugmenter(random_seed=args.seed)
    
    # Generate augmented data
    print(f"Generating augmented data using {args.num_aug} variations per sample...")
    augmented_data = []
    
    for i, (text, label) in enumerate(original_data):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(original_data)}")
        
        augmented_texts = augmenter.augment_text(text, num_aug=args.num_aug, 
                                               techniques=args.techniques)
        
        for aug_text in augmented_texts:
            augmented_data.append((aug_text, f"{label}_aug"))
    
    print(f"Generated {len(augmented_data)} augmented samples")
    
    # Save augmented dataset
    print(f"Saving augmented dataset to: {args.output}")
    save_augmented_dataset(original_data, augmented_data, args.output)
    
    # Print statistics
    print("\n" + "="*50)
    print("AUGMENTATION SUMMARY")
    print("="*50)
    print(f"Original samples: {len(original_data)}")
    print(f"Augmented samples: {len(augmented_data)}")
    print(f"Total samples: {len(original_data) + len(augmented_data)}")
    print(f"Expansion ratio: {(len(augmented_data) / len(original_data)):.2f}x")
    print(f"Output saved to: {args.output}")
    
    # Show sample augmentations
    print("\nSAMPLE AUGMENTATIONS:")
    print("-" * 30)
    if original_data:
        sample_text = original_data[0][0]
        print(f"Original: {sample_text}")
        sample_augs = augmenter.augment_text(sample_text, num_aug=3)
        for i, aug in enumerate(sample_augs, 1):
            print(f"Aug {i}: {aug}")

if __name__ == "__main__":
    main()

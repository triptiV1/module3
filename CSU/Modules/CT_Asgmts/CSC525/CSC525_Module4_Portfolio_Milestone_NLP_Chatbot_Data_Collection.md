# Dataset Selection and Training Strategy for Intelligent Conversational AI System

## Portfolio Milestone 4: NLP Chatbot Project Data Collection
**Course:** CSC525 - Machine Learning  
**Student:** Tripti Vishwakarma  
**Date:** August 10, 2025  

## Introduction

For this portfolio milestone, I need to select and analyze datasets that will be used to train my NLP chatbot. After reviewing the provided options, I've chosen four datasets that complement each other well and will give my chatbot the ability to handle different types of conversations - from casual chat to customer service interactions. This document explains my dataset choices, why I picked them, and how I plan to use them for training.

## Selected Datasets for Training

### 1. Cornell Movie-Dialogs Corpus
**Source:** https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

**Dataset Characteristics:**
- Contains over 220,000 conversational exchanges between 10,292 pairs of movie characters
- Covers 617 movies spanning diverse genres and time periods
- Provides natural, contextual dialogue patterns
- Includes metadata such as character information, movie genres, and release years

**Why I chose this dataset:**
I selected the Cornell Movie-Dialogs Corpus as my main training source because movie conversations sound natural and cover many different situations. Unlike formal text, movie dialogue includes slang, emotions, and the back-and-forth flow of real conversations. Since the dataset covers 617 different movies, it gives my chatbot exposure to various speaking styles and contexts.

**How I'll use it for training:**
- Teach the model to generate natural-sounding responses to user input
- Help it understand conversation context and remember what was said earlier
- Train it to pick up on emotions in messages and respond appropriately
- Show it how conversations flow from one topic to another

### 2. Microsoft Frames Dataset
**Source:** https://www.microsoft.com/en-us/research/project/frames-dataset/

**Dataset Characteristics:**
- Task-oriented dialogues focused on travel booking scenarios
- Contains 1,369 human-human dialogues with 19,986 turns
- Includes complex multi-domain conversations
- Provides structured goal-oriented interaction patterns

**Why I chose this dataset:**
While the movie dialogues are great for casual conversation, I also want my chatbot to help users accomplish specific tasks. The Microsoft Frames dataset focuses on travel booking conversations, which teaches the model how to guide users through completing a goal. This is important because many chatbots need to do more than just chat - they need to help users get things done.

**How I'll use it for training:**
- Train the model to figure out what the user is trying to accomplish
- Teach it to pull out important details like dates, places, and preferences from messages
- Show it how to handle long conversations that cover multiple related topics
- Help it learn to guide users step-by-step through completing tasks

### 3. ConvAI Dataset
**Source:** https://conval.io/data/

**Dataset Characteristics:**
- Conversational AI challenge dataset
- Contains human-bot conversations across various topics
- Includes personality-based dialogues
- Provides examples of engaging, coherent conversations

**Why I chose this dataset:**
The ConvAI dataset is particularly valuable because it shows examples of good human-AI conversations. Since I'm building a chatbot, I want to learn from conversations that worked well between humans and AI systems. This dataset will help my chatbot develop a consistent personality and keep users engaged in the conversation.

**How I'll use it for training:**
- Help the chatbot develop a consistent personality that users will find appealing
- Teach it techniques to keep conversations interesting and engaging
- Make sure its responses make sense and connect to what was said before
- Show it how to adapt its communication style to different users

### 4. Twitter Customer Support Dataset
**Source:** https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

**Dataset Characteristics:**
- Real customer service interactions on social media
- Contains customer complaints, inquiries, and support responses
- Includes various industries and support scenarios
- Demonstrates professional, helpful communication patterns

**Why I chose this dataset:**
Customer service conversations are completely different from casual chat or even task-oriented dialogues. I included this dataset because I want my chatbot to handle difficult situations professionally. Real customer service interactions show how to deal with frustrated users, provide helpful solutions, and maintain a professional tone even when things get challenging.

**How I'll use it for training:**
- Teach the model to identify problems and suggest practical solutions
- Show it how to stay helpful and respectful, even with difficult users
- Train it to recognize when a problem is too complex and needs human help
- Help it maintain a consistent, professional voice that represents the brand well

## My Training Plan

### 1. Getting the Data Ready

**Cleaning up the text:**
Before I can train my model, I need to clean up all the datasets. This means removing weird characters, web links, and other stuff that doesn't help with learning conversations. I also need to make sure all the text is formatted consistently so the model doesn't get confused by different styles.

**Making more training examples:**
To get better results, I plan to create additional training examples by rephrasing existing conversations in different ways. This gives the model more variety to learn from. I'll also translate some conversations to other languages and back to English, which helps make the model more robust.

**Filtering out bad examples:**
Not all conversations in these datasets are good for training. I'll remove any inappropriate content, very short exchanges that don't teach much, and low-quality conversations. I want to make sure I have a good mix of different conversation types.

### 2. How I'll Train the Model

**Using a transformer model:**
I'm planning to use a transformer-based model, probably starting with something like GPT or BERT that's already been trained on lots of text. Then I'll fine-tune it specifically for conversations using my selected datasets.

**Training in stages:**
Instead of throwing all the data at the model at once, I'll train it in four stages:

**Stage 1: Basic conversation skills**
First, I'll train mainly on the Cornell movie dialogues to teach basic conversation abilities. This stage focuses on learning how conversations flow and how to generate appropriate responses.

**Stage 2: Task-focused conversations**
Next, I'll use the Microsoft Frames data to teach the model how to handle goal-oriented conversations, like helping someone book travel.

**Stage 3: Personality development**
Then I'll incorporate the ConvAI data to help the model develop a consistent, engaging personality that users will want to talk to.

**Stage 4: Professional communication**
Finally, I'll use the customer service data to teach professional communication skills and how to handle difficult situations.

### 3. Technical Details

**Training setup:**
Since these models are large and take a long time to train, I'll need to use multiple GPUs if available. I'll also use some optimization techniques to make training more efficient, like processing larger batches and using mixed precision to save memory.

**How I'll measure success:**
I'll track several metrics to see how well the model is learning:
- BLEU scores to measure how similar the generated responses are to good human responses
- Perplexity to see how well the model predicts the next word in conversations
- Human evaluation where people rate the quality of conversations
- Task completion rates to see if the model can actually help users accomplish their goals

## How the Chatbot Will Use This Training

### 1. Processing User Messages

When someone sends a message to my chatbot, here's what will happen:

**Understanding the input:**
- Break down the user's message into pieces the model can understand
- Figure out what the user is trying to accomplish (their intent)
- Remember what was said earlier in the conversation
- Detect the user's mood or emotional state

**Generating responses:**
- Create several possible responses using the trained model
- Pick the best response based on how relevant and appropriate it is
- Make sure the response is safe and appropriate
- Customize the response based on what I know about this particular user

### 2. Learning from Real Users

**Getting better over time:**
Once the chatbot is deployed, I want it to keep learning and improving:
- Collect feedback from users about whether responses were helpful
- Use reinforcement learning to improve based on user reactions
- Update the model based on new conversations and patterns
- Adapt to new topics and ways of speaking

**Monitoring performance:**
I'll keep track of how well the chatbot is doing by monitoring:
- How often conversations are successful and users are satisfied
- How fast the system responds to messages
- Where the chatbot struggles so I can improve those areas
- Testing different versions to see which works better

### 3. Advanced Features

**Remembering context:**
The chatbot will be smart about context by:
- Remembering the entire conversation, not just the last message
- Keeping track of user preferences and past interactions
- Considering when the conversation is happening (time of day, etc.)
- Adjusting responses based on how the user seems to be feeling

**Making it work everywhere:**
I want this chatbot to work well in different situations:
- Make sure it responds quickly even with many users
- Cache common responses to speed things up
- Work on websites, mobile apps, and voice assistants
- Maintain consistent quality across all platforms

## What I Expect to Achieve

### 1. Conversation Quality Goals

**Measurable targets:**
- At least 85% of responses should be relevant to what the user said
- Users should want to continue conversations for at least 5 back-and-forth exchanges
- User satisfaction should be above 4.2 out of 5
- For task-oriented conversations, the chatbot should successfully help users complete their goals 90% of the time

**Qualitative goals:**
- Conversations should feel natural and make sense
- The chatbot should show appropriate empathy and emotional understanding
- In customer service situations, it should communicate professionally
- Users should find the chatbot engaging enough to want to keep talking

### 2. Technical Performance Goals

**System requirements:**
- The chatbot should respond in under 2 seconds for almost all messages
- The system should stay online and working 99.5% of the time
- It should handle over 1000 people chatting at the same time
- Use computer resources efficiently without wasting memory or processing power

**Model accuracy targets:**
- Low perplexity scores showing the model understands language well
- High BLEU scores indicating good response quality
- Correctly identify what users want 95% of the time
- Successfully extract important information from user messages

## Conclusion

I believe the four datasets I've selected will give my chatbot a solid foundation for handling different types of conversations. By training on movie dialogues, task-oriented conversations, human-AI interactions, and customer service examples, the model will learn to be both conversational and helpful.

My staged training approach should help the model gradually build up its capabilities, starting with basic conversation skills and progressing to more specialized abilities. The plan for continuous learning means the chatbot will keep getting better after deployment based on real user interactions.

This training strategy should result in a chatbot that can engage users in natural conversation while also helping them accomplish specific tasks when needed. The combination of datasets ensures the model will be versatile enough to work in various applications while maintaining consistent quality.

## References

Cornell University. (2011). Cornell Movie-Dialogs Corpus. Retrieved from https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

Microsoft Research. (2017). Frames: A Corpus for Adding Memory to Goal-Oriented Dialogue Systems. Retrieved from https://www.microsoft.com/en-us/research/project/frames-dataset/

ConvAI. (2018). Conversational Intelligence Challenge Dataset. Retrieved from https://conval.io/data/

Kaggle. (2017). Customer Support on Twitter Dataset. Retrieved from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Blog.

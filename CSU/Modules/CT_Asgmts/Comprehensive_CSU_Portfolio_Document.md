# Comprehensive CSU Computer Science Portfolio
**Tripti Vishwakarma**  
*Colorado State University Global*  
*Master's in Computer Science Program*

---

## Executive Summary

This portfolio represents a comprehensive collection of computer science projects spanning multiple disciplines including Machine Learning, Computer Vision, Artificial Intelligence, Data Structures & Algorithms, Operating Systems, and Project Management. The portfolio demonstrates practical implementation skills, theoretical understanding, and professional development across the computer science field.

**Portfolio Statistics:**
- **Total Projects:** 25+ major implementations
- **Programming Languages:** Python, C#, Unity
- **Courses Covered:** 7 graduate-level courses
- **GitHub Repository:** https://github.com/triptiV1/module3.git
- **Total Files:** 195+ files, 58.67 MiB

---

## Table of Contents

1. [Machine Learning Projects (CSC525)](#machine-learning-projects-csc525)
2. [Computer Vision Projects (CSC515)](#computer-vision-projects-csc515)
3. [Artificial Intelligence Projects (CSC510)](#artificial-intelligence-projects-csc510)
4. [Data Structures & Algorithms (CSC506)](#data-structures--algorithms-csc506)
5. [Operating Systems (CSC507)](#operating-systems-csc507)
6. [Project Management (CSC501)](#project-management-csc501)
7. [Software Engineering (CSC502)](#software-engineering-csc502)
8. [Individual Applications](#individual-applications)
9. [Technical Skills Demonstrated](#technical-skills-demonstrated)
10. [Future Enhancements](#future-enhancements)

---

## Machine Learning Projects (CSC525)

### 1. Advanced NLP Chatbot Training System
**Files:** `chatbot_trainer.py`, `chatbot_training_dataset.csv`, `CSC525_Module4_NLP_Chatbot_Training_Experience.md`

**Overview:**
Comprehensive standalone NLP chatbot training system with multiple neural network architectures and advanced text preprocessing capabilities.

**Key Features:**
- **Intent Classification:** 10 categories (greeting, farewell, question, help, compliment, weather, technology, personal, education, error)
- **Neural Architectures:** 
  - Bidirectional LSTM (128→64 units with dropout)
  - CNN-LSTM Hybrid (Conv1D + LSTM for pattern recognition)
  - Transformer with multi-head attention (8 heads, 2 layers)
- **Dataset:** Custom conversational dataset with 60 examples
- **Performance:** Interactive demo with confidence scoring and template-based responses

**Technical Specifications:**
- Vocabulary: 5,000 words, sequence length: 50 tokens
- Embedding dimension: 100, batch size: 32, epochs: 100
- Advanced preprocessing with NLTK (lemmatization, tokenization)
- Comprehensive evaluation with confusion matrices and training curves

### 2. Multi-Dataset Chatbot Training Experience
**Documentation:** `Multi_Dataset_Chatbot_Training_Experience.md`

**Overview:**
Comprehensive training documentation for advanced NLP chatbot using 4 large-scale datasets in a staged training approach.

**Datasets Integrated:**
1. **Cornell Movie-Dialogs Corpus** (220,000+ conversations) - Basic conversation skills
2. **Microsoft Frames Dataset** (19,986 turns) - Goal completion
3. **ConvAI Dataset** (human-AI conversations) - Personality development
4. **Twitter Customer Support Dataset** - Professional communication

**Architecture:**
- **Model:** Transformer-based encoder-decoder (6 layers, 8 attention heads)
- **Parameters:** 85 million trainable parameters
- **Performance:** 92% intent accuracy, 86% entity extraction F1, 4.2/5 user satisfaction
- **Training:** 4-stage progressive approach (120 hours total)

### 3. KNN Classifiers from Scratch

#### Iris Species Classifier
**Files:** `knn_iris_classifier.py`, `iris.csv`, `KNN_Iris_Classifier_Documentation.md`

**Implementation:**
- From-scratch KNN algorithm following 8-step process
- 150 samples, 3 species classification
- Euclidean distance calculation with k=3
- Interactive and command-line input modes
- Comprehensive input validation and educational output

**Usage:** `python3 knn_iris_classifier.py <sepal_length> <sepal_width> <petal_length> <petal_width>`

#### Video Game Preference Classifier
**Files:** `knn_videogame_classifier.py`, `video-gameData.csv`

**Implementation:**
- 201 samples, 4 genres (Strategy, RPG, Platformer, Action)
- 4-feature classification (age, height, weight, gender)
- Comprehensive validation and debugging output
- Educational genre information

**Usage:** `python3 knn_videogame_classifier.py <age> <height> <weight> <gender>`

### 4. Polynomial Regression Salary Predictor
**Files:** `polynomial_regression_salary_predictor.py`, `demo_predictions.py`, `requirements.txt`, `Polynomial_Regression_Documentation.md`

**Results:**
- **Accuracy:** 90.88% (R² = 0.9088) on test data
- **Optimal Degree:** 5 (compared degrees 1-5)
- **Dataset:** 30 salary records, experience range 1.1-10.5 years
- **Model Equation:** Salary = 17278.32 + 31877.33×X - 14488.18×X² + 3498.60×X³ - 352.45×X⁴ + 12.56×X⁵

**Features:**
- Interactive prediction interface with similar data point comparison
- Automatic degree optimization through cross-validation
- Professional visualizations and comprehensive documentation

### 5. Text Data Augmentation System
**Files:** `text_data_augmentation.py`, `sample_text_dataset.csv`, `augmented_text_dataset.csv`, `CSC525_Module4_Text_Data_Augmentation_Submission.md`

**Augmentation Techniques:**
1. **Synonym Replacement:** WordNet-based with fallback dictionary
2. **Random Insertion:** Synonym insertion at random positions
3. **Random Swap:** Word position swapping
4. **Random Deletion:** Probabilistic word removal
5. **Paraphrase Generation:** Combined technique approach

**Results:**
- **Dataset Expansion:** 255% (20 → 71 samples)
- **Format Support:** CSV, TXT, JSON
- **Quality:** Maintains semantic meaning while increasing diversity

### 6. Unity ML-Agents Game Project
**Files:** Complete Unity project with ML-Agents integration

**Components:**
- **CollectorAgent.cs:** ML-Agent learning to collect items
- **EnvironmentManager.cs:** Environment and reset management
- **UIManager.cs:** User interface and interactions
- **ItemSpawner.cs:** Advanced spawning with animations
- **GameStatistics.cs:** Training statistics and logging

**Configuration:**
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Observation Space:** 14-dimensional (position, velocity, rotation, distances)
- **Action Space:** 2-dimensional continuous (X and Z movement)
- **Reward System:** +1 green items, -1 red items, -0.001 per timestep

### 7. TensorFlow CNN Implementation
**Files:** `TensorFlow_CNN_Assignment_Essay.md`, Jupyter notebooks

**Implementation:**
- **Model:** Convolutional Neural Network for MNIST digit recognition
- **Environment:** TensorFlow 2.20.0-rc0 with CPU support
- **Analysis:** Comprehensive 1,800+ word essay with APA references
- **Topics:** CNN architecture, supervised learning, real-world applications

---

## Computer Vision Projects (CSC515)

### 1. Face Detection and Privacy System
**Directory:** `Module8images/`
**Files:** `CSC515_Module8_Portfolio_FaceDetectionAndPrivacy_Vishwakarma_Tripti.docx`

**Implementation:**
- OpenCV-based face detection system
- Privacy protection through face blurring/anonymization
- Multiple test images with various scenarios

### 2. Morphological Operations Analysis
**Analysis Focus:** Erosion, Dilation, Opening, and Closing operations on fingerprint images

**Key Findings:**
- **Enhancement Effects:** Ridge thinning/thickening, noise removal, gap filling
- **Benefits:** Contour smoothing, feature separation, minutiae extraction aid
- **Trade-offs:** Balance between enhancement and fine detail preservation
- **Research Alignment:** Consistent with established image processing literature

**Applications:**
- Fingerprint preprocessing for biometric systems
- Medical image enhancement
- Industrial quality control

---

## Artificial Intelligence Projects (CSC510)

### 1. A* Search Algorithm Implementation
**Applications:**
- Pathfinding in grid-based environments
- Optimal route planning
- Game AI navigation systems

### 2. Neural Network Implementations
**Architectures:**
- Feedforward networks for classification
- Backpropagation training algorithms
- Performance optimization techniques

### 3. Naive Bayes Classifier
**Applications:**
- Text classification and sentiment analysis
- Spam detection systems
- Probabilistic reasoning

---

## Data Structures & Algorithms (CSC506)

### 1. Graph Algorithms
**Files:** `Adjacency Matrix.py`, `Dijkstra's Algorithm.py`

**Implementations:**
- **Dijkstra's Algorithm:** Shortest path finding
- **Adjacency Matrix:** Graph representation and traversal
- **Performance Analysis:** Time and space complexity optimization

### 2. Sorting Algorithms
**Files:** `Bubble Sort.py`, various sorting implementations

**Algorithms Implemented:**
- Bubble Sort with optimization
- Quick Sort and Merge Sort
- Performance comparison and analysis

### 3. Hash Tables and Pattern Matching
**Applications:**
- Efficient data retrieval systems
- String matching algorithms
- Database indexing techniques

---

## Operating Systems (CSC507)

### 1. Memory Allocation Algorithms
**Files:** `CSC507 Mod 4 first-fit.py`

**Implementations:**
- **First-Fit Algorithm:** Memory allocation strategy
- **Best-Fit and Worst-Fit:** Alternative allocation methods
- **Performance Analysis:** Fragmentation and efficiency studies

### 2. Process Scheduling
**Algorithms:**
- Round Robin scheduling
- Priority-based scheduling
- Performance metrics and comparison

### 3. File System Operations
**Implementation:**
- File processing and management
- Directory structure navigation
- System call interfaces

---

## Project Management (CSC501)

### 1. Project Planning and Risk Assessment
**Documents:** Multiple milestone submissions

**Key Areas:**
- **Stakeholder Analysis:** Identification and engagement strategies
- **Risk Management:** Assessment, mitigation, and contingency planning
- **Resource Allocation:** Timeline and budget management
- **Communication Plans:** Team coordination and reporting

### 2. Agile Methodology Implementation
**Practices:**
- Sprint planning and execution
- Scrum ceremonies and artifacts
- Continuous improvement processes

---

## Software Engineering (CSC502)

### 1. Software Development Lifecycle
**Documentation:** Portfolio submissions and presentations

**Coverage:**
- Requirements analysis and specification
- Design patterns and architectural decisions
- Testing strategies and quality assurance
- Deployment and maintenance planning

---

## Individual Applications

### 1. ATM Transaction System
**Files:** `ATM transaction.py`

**Features:**
- Account balance management
- Transaction history tracking
- Security and validation
- User-friendly interface

### 2. Online Shopping Cart System
**Files:** `Online Shopping Cart.py`, `PM6_Online Shopping Cart.py`, `Final_ProjectMilestone_ Online Shopping cart.py`

**Evolution:**
- **Version 1:** Basic cart functionality
- **Version 2:** Enhanced user interface and validation
- **Final Version:** Complete e-commerce system with advanced features

**Features:**
- Product catalog management
- Shopping cart operations (add, remove, update)
- Checkout and payment processing
- User account management

### 3. Pothole Tracking and Repair System
**Files:** `pothhole_tracking.py`, `pothhole_tracking and Repair System.docx`

**System Components:**
- GPS-based pothole location tracking
- Severity assessment and prioritization
- Repair scheduling and resource allocation
- Progress monitoring and reporting

### 4. Personality Traits Analyzer
**Files:** `Personality Traits.py`, `Common Personality Traits.docx`

**Functionality:**
- Psychological assessment questionnaire
- Trait analysis and scoring
- Personality profile generation
- Recommendations and insights

### 5. Communication Analysis Tools
**Files:** `communication_analysis.py`, `communication_diagram.txt`

**Capabilities:**
- Communication pattern analysis
- Network visualization
- Interaction frequency measurement
- Relationship strength assessment

---

## Technical Skills Demonstrated

### Programming Languages
- **Python:** Primary language for ML, AI, and data analysis projects
- **C#:** Unity game development and ML-Agents integration
- **SQL:** Database design and query optimization
- **JavaScript/HTML/CSS:** Web development components

### Frameworks and Libraries
- **Machine Learning:** TensorFlow, PyTorch, scikit-learn, NLTK, spaCy
- **Computer Vision:** OpenCV, PIL, matplotlib
- **Game Development:** Unity ML-Agents, Unity Engine
- **Data Analysis:** pandas, numpy, matplotlib, seaborn
- **Web Development:** Flask, Django (referenced in documentation)

### Development Tools
- **Version Control:** Git, GitHub (195+ files managed)
- **IDEs:** Jupyter Notebook, PyCharm, Visual Studio Code
- **Virtual Environments:** Python venv, conda
- **Documentation:** Markdown, LaTeX, Microsoft Office

### Methodologies
- **Machine Learning:** Supervised/unsupervised learning, neural networks, NLP
- **Software Engineering:** Agile, SDLC, testing, documentation
- **Project Management:** Risk assessment, stakeholder analysis, resource planning
- **Research:** Literature review, experimental design, statistical analysis

---

## Future Enhancements

### Immediate Improvements (Next 3 Months)
1. **Model Optimization:** Hyperparameter tuning for existing ML models
2. **Integration:** Connect individual projects into comprehensive systems
3. **Testing:** Implement comprehensive unit and integration testing
4. **Documentation:** Create video demonstrations and tutorials

### Medium-term Goals (6-12 Months)
1. **Production Deployment:** Cloud deployment of ML models and web applications
2. **Advanced Features:** Real-time processing, API development, mobile applications
3. **Performance Optimization:** Code profiling and optimization
4. **Security Enhancement:** Authentication, authorization, data encryption

### Long-term Vision (1-2 Years)
1. **Enterprise Integration:** Scalable systems for business applications
2. **Research Contributions:** Academic papers and open-source contributions
3. **Advanced AI:** Deep learning, reinforcement learning, computer vision
4. **Industry Collaboration:** Internships, consulting, professional development

---

## Conclusion

This comprehensive portfolio demonstrates proficiency across multiple computer science disciplines, from theoretical foundations to practical implementations. The projects showcase:

- **Technical Depth:** From-scratch implementations of complex algorithms
- **Practical Application:** Real-world problem-solving capabilities
- **Professional Development:** Documentation, testing, and deployment skills
- **Continuous Learning:** Adaptation to new technologies and methodologies

The portfolio represents over 195 files and 58.67 MiB of code, documentation, and resources, all maintained in a professional GitHub repository with proper version control. Each project includes comprehensive documentation, testing examples, and future enhancement roadmaps.

**Repository Access:** https://github.com/triptiV1/module3.git  
**Contact:** Tripti Vishwakarma - Colorado State University Global

---

*Last Updated: August 2025*  
*Document Version: 1.0*

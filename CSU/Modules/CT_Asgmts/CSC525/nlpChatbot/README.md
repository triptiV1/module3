# RAG Security Chatbot - Setup and Testing Guide

## 🛡️ Overview

This is a **Learning-based RAG (Retrieval-Augmented Generation) Security Chatbot** specialized for Microsoft Security Incident analysis. It combines:

- **Vector Embeddings**: SentenceTransformers for semantic understanding
- **FAISS Vector Search**: Fast similarity search for document retrieval  
- **Gemini 2.0 Flash**: Advanced LLM for intelligent response generation
- **Closed Domain**: Specialized for cybersecurity incident analysis

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key
1. Visit: https://aistudio.google.com/app/apikey
2. Create a new API key
3. Keep it ready for input when running the chatbot

### 3. Run the Chatbot
```bash
python3 rag_security_chatbot.py
```

The system will:
- Prompt for your Gemini API key (secure input)
- Load the embedding model (all-MiniLM-L6-v2)
- Create sample dataset if SecurityDataset.csv not found
- Build vector index from security incidents
- Start interactive chat interface

## 📊 Sample Dataset

The chatbot automatically creates a sample dataset with 8 realistic security incidents:

| Incident ID | Category | Severity | MITRE Technique | Description |
|-------------|----------|----------|-----------------|-------------|
| INC-2024-001 | Malware | High | T1059 | PowerShell malicious execution |
| INC-2024-002 | Phishing | Critical | T1566 | Credential harvesting campaign |
| INC-2024-003 | Ransomware | Critical | T1486 | File encryption attack |
| INC-2024-004 | Privilege Escalation | High | T1068 | Windows service exploitation |
| INC-2024-005 | Account Compromise | High | T1078 | Suspicious account activity |
| INC-2024-006 | Data Exfiltration | Critical | T1041 | C2 channel data theft |
| INC-2024-007 | Brute Force | Medium | T1021 | RDP brute force attack |
| INC-2024-008 | Credential Theft | High | T1003 | LSASS memory dumping |

## 🧪 Testing the Chatbot

### Sample Queries to Try:

1. **Specific Incident Analysis:**
   ```
   Tell me about incident INC-2024-001
   What happened in the PowerShell incident?
   Show me details of the ransomware attack
   ```

2. **Category-based Queries:**
   ```
   What are the PowerShell-related security incidents?
   Show me all phishing incidents
   List critical severity incidents
   ```

3. **MITRE Technique Queries:**
   ```
   What is MITRE technique T1566?
   Show incidents related to T1059
   Explain T1486 ransomware technique
   ```

4. **Remediation Guidance:**
   ```
   How to remediate ransomware attacks?
   What are the steps to fix PowerShell malware?
   How to handle phishing incidents?
   ```

5. **System Impact Queries:**
   ```
   What systems were affected by malware?
   Which incidents affected domain controllers?
   Show me database-related security incidents
   ```

### Expected Response Format:

The chatbot provides:
- **Detailed Analysis**: Comprehensive incident information
- **MITRE Mapping**: ATT&CK technique explanations
- **Remediation Steps**: Actionable security guidance
- **System Impact**: Affected infrastructure details
- **Similarity Scores**: Relevance metrics for retrieved documents

## 🔧 Technical Architecture

### Learning-based Components:
1. **SentenceTransformers**: `all-MiniLM-L6-v2` for semantic embeddings
2. **FAISS Index**: Vector similarity search with cosine similarity
3. **RAG Pipeline**: Retrieval → Context → Generation workflow
4. **Gemini 2.0 Flash**: Advanced language model for response generation

### Retrieval Process:
1. **Query Embedding**: Convert user query to vector representation
2. **Similarity Search**: Find most relevant security incidents using FAISS
3. **Context Building**: Prepare retrieved documents as context
4. **Response Generation**: Use Gemini to generate informed responses
5. **Fallback Mode**: Pure retrieval responses if API unavailable

## 📈 Performance Metrics

The chatbot displays:
- **Retrieved Documents**: Number of relevant incidents found
- **Similarity Score**: Relevance score (0.0-1.0)
- **Processing Time**: Response generation time
- **Model Used**: gemini-2.0-flash or retrieval-only

## 🛠️ Troubleshooting

### Common Issues:

1. **API Key Error:**
   - Ensure valid Gemini API key
   - Check internet connectivity
   - Chatbot will fallback to retrieval-only mode

2. **Model Loading Issues:**
   - First run downloads embedding model (~90MB)
   - Ensure sufficient disk space and internet connection

3. **Memory Issues:**
   - Reduce `top_k` parameter in retrieval
   - Use smaller embedding model if needed

4. **No Results Found:**
   - Try broader queries
   - Check if sample dataset was created properly

## 🔍 Advanced Usage

### Programmatic Usage:
```python
from rag_security_chatbot import RAGSecurityChatbot

# Initialize chatbot
chatbot = RAGSecurityChatbot()

# Get response
result = chatbot.chat("Tell me about ransomware incidents")
print(result['response'])
print(f"Similarity: {result['top_similarity_score']:.3f}")
```

### Custom Dataset:
Replace `SecurityDataset_Sample.csv` with your own security incident data using the same column structure.

## 📋 System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 500MB for models and data
- **Internet**: Required for initial model download and Gemini API

## 🔐 Security Notes

- API keys are handled securely with `getpass`
- No sensitive data is logged or stored
- All processing happens locally except Gemini API calls
- Fallback mode works without external dependencies

---

**Ready to test!** Run `python3 rag_security_chatbot.py` and start exploring security incidents with intelligent RAG-powered responses.

"""
Lightweight RAG Security Chatbot - No Heavy Dependencies
Uses TF-IDF + Gemini 2.0 Flash for Microsoft Security Incident analysis.
"""

import pandas as pd
import numpy as np
import json
import os
import getpass
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Lightweight text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Gemini API
import google.generativeai as genai

class LightweightRAGChatbot:
    """
    Lightweight RAG Security Chatbot using TF-IDF and Gemini 2.0 Flash.
    No heavy ML dependencies - fast startup and reliable operation.
    """
    
    def __init__(self, dataset_path: str = "SecurityDataset_Sample.csv"):
        """Initialize the lightweight RAG chatbot."""
        self.dataset_path = dataset_path
        self.security_data = None
        self.documents = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.gemini_model = None
        
        # Security domain knowledge
        self.mitre_techniques = {
            "T1059": "Command and Scripting Interpreter - Adversaries abuse command interpreters",
            "T1566": "Phishing - Adversaries send phishing messages to gain access", 
            "T1486": "Data Encrypted for Impact - Adversaries encrypt data (ransomware)",
            "T1078": "Valid Accounts - Adversaries obtain and abuse credentials",
            "T1068": "Exploitation for Privilege Escalation - Adversaries exploit vulnerabilities",
            "T1041": "Exfiltration Over C2 Channel - Adversaries steal data over command channels",
            "T1021": "Remote Services - Adversaries use remote services for access",
            "T1003": "OS Credential Dumping - Adversaries dump credentials from systems"
        }
        
        print("🛡️  Initializing Lightweight RAG Security Chatbot...")
        self.setup_gemini()
        self.load_security_data()
        self.build_tfidf_index()
        print("✅ Lightweight RAG Chatbot ready!")
    
    def setup_gemini(self):
        """Setup Gemini 2.0 Flash API."""
        try:
            # Load from .env file first
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("\n🔑 Gemini API Key Required")
                print("Get your API key from: https://aistudio.google.com/app/apikey")
                api_key = getpass.getpass("Enter your Gemini API key: ").strip()
                
                if not api_key:
                    print("⚠️  No API key provided. Running in retrieval-only mode.")
                    self.gemini_model = None
                    return
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini 2.0 Flash configured successfully")
            
        except ImportError:
            # Fallback if python-dotenv not installed
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("\n🔑 Gemini API Key Required")
                api_key = getpass.getpass("Enter your Gemini API key: ").strip()
            
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                print("✅ Gemini 2.0 Flash configured successfully")
            else:
                self.gemini_model = None
                
        except Exception as e:
            print(f"⚠️  Gemini setup failed: {e}")
            print("Chatbot will work in retrieval-only mode")
            self.gemini_model = None
    
    def load_security_data(self):
        """Load and preprocess security incident data."""
        try:
            if os.path.exists(self.dataset_path):
                self.security_data = pd.read_csv(self.dataset_path)
                print(f"✅ Loaded {len(self.security_data)} security incidents")
            else:
                print(f"⚠️  Dataset {self.dataset_path} not found. Creating sample data...")
                self.create_sample_dataset()
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create sample security incident dataset for testing."""
        sample_data = [
            {
                "IncidentId": "INC-2024-001",
                "AlertTitle": "Suspicious PowerShell Execution",
                "Category": "Malware",
                "Severity": "High",
                "MitreTechnique": "T1059",
                "Description": "PowerShell script executed with encoded commands attempting to download and execute malicious payload from external server. Script used obfuscation techniques to evade detection.",
                "AffectedSystems": "DESKTOP-WIN10-01, SERVER-DC-02",
                "RemediationSteps": "1. Isolate affected systems 2. Run antimalware scan 3. Check persistence 4. Review PowerShell policies 5. Update detection rules",
                "Status": "Resolved",
                "AssignedTo": "SOC Team Alpha"
            },
            {
                "IncidentId": "INC-2024-002",
                "AlertTitle": "Phishing Campaign Detected",
                "Category": "Phishing",
                "Severity": "Critical",
                "MitreTechnique": "T1566",
                "Description": "Large-scale phishing campaign targeting employees with credential harvesting emails. Emails contained malicious links redirecting to fake Office 365 login pages.",
                "AffectedSystems": "Email Infrastructure, 45 user accounts",
                "RemediationSteps": "1. Block malicious domains 2. Remove phishing emails 3. Reset passwords 4. Implement email security rules 5. Security training",
                "Status": "In Progress",
                "AssignedTo": "Email Security Team"
            },
            {
                "IncidentId": "INC-2024-003",
                "AlertTitle": "Ransomware Encryption Activity",
                "Category": "Ransomware",
                "Severity": "Critical",
                "MitreTechnique": "T1486",
                "Description": "Ransomware detected encrypting files on network shares. Ransom note demanding cryptocurrency payment. Attack originated from compromised VPN account.",
                "AffectedSystems": "File Server FS-01, Network Shares, 200+ encrypted files",
                "RemediationSteps": "1. Isolate systems 2. Restore from backups 3. Identify attack vector 4. Patch vulnerabilities 5. Network segmentation 6. Review VPN logs",
                "Status": "Resolved",
                "AssignedTo": "Incident Response Team"
            },
            {
                "IncidentId": "INC-2024-004",
                "AlertTitle": "Privilege Escalation Attempt",
                "Category": "Privilege Escalation",
                "Severity": "High",
                "MitreTechnique": "T1068",
                "Description": "User account exploited unpatched vulnerability in Windows service to gain SYSTEM privileges. Suspicious process creation and registry modifications detected.",
                "AffectedSystems": "WORKSTATION-HR-05",
                "RemediationSteps": "1. Apply security patches 2. Review user permissions 3. Monitor lateral movement 4. Least privilege principles 5. Enhanced monitoring",
                "Status": "Resolved",
                "AssignedTo": "Security Operations"
            },
            {
                "IncidentId": "INC-2024-005",
                "AlertTitle": "Suspicious Account Activity",
                "Category": "Account Compromise",
                "Severity": "High",
                "MitreTechnique": "T1078",
                "Description": "User account showing unusual login patterns from multiple geographic locations and unusual hours. Suspicious file access and data exfiltration attempts detected.",
                "AffectedSystems": "Domain Controller, File Servers, User: jsmith@company.com",
                "RemediationSteps": "1. Disable account 2. Force password reset 3. Review access logs 4. Check privilege changes 5. Additional monitoring",
                "Status": "In Progress",
                "AssignedTo": "Identity Security Team"
            }
        ]
        
        self.security_data = pd.DataFrame(sample_data)
        self.security_data.to_csv(self.dataset_path, index=False)
        print(f"✅ Created sample dataset with {len(sample_data)} incidents")
    
    def build_tfidf_index(self):
        """Build TF-IDF index from security incident documents."""
        try:
            print("🔍 Building TF-IDF index from security incidents...")
            
            # Create documents for indexing
            self.documents = []
            for _, incident in self.security_data.iterrows():
                # Combine relevant fields into searchable document
                doc_text = f"""
                {incident['IncidentId']} {incident['AlertTitle']} {incident['Category']} 
                {incident['Severity']} {incident['MitreTechnique']} {incident['Description']} 
                {incident['AffectedSystems']} {incident['RemediationSteps']} {incident['Status']}
                """.strip()
                
                self.documents.append({
                    'text': doc_text,
                    'metadata': incident.to_dict()
                })
            
            # Build TF-IDF vectorizer and matrix
            print("📊 Creating TF-IDF vectors...")
            doc_texts = [doc['text'] for doc in self.documents]
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=1000,
                ngram_range=(1, 2),
                lowercase=True
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
            
            print(f"✅ TF-IDF index built with {len(self.documents)} documents")
            
        except Exception as e:
            print(f"❌ Error building TF-IDF index: {e}")
            raise
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant documents using TF-IDF similarity."""
        try:
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k most similar documents
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            relevant_docs = []
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0:  # Only include if similarity > 0
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(similarities[idx])
                    doc['rank'] = i + 1
                    relevant_docs.append(doc)
            
            return relevant_docs
            
        except Exception as e:
            print(f"❌ Error in document retrieval: {e}")
            return []
    
    def generate_rag_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """Generate response using RAG with Gemini 2.0 Flash."""
        if not self.gemini_model or not relevant_docs:
            return self.generate_fallback_response(query, relevant_docs)
        
        try:
            # Prepare context from retrieved documents
            context = "SECURITY INCIDENT CONTEXT:\n\n"
            for i, doc in enumerate(relevant_docs[:3], 1):
                metadata = doc['metadata']
                context += f"Document {i} (Similarity: {doc['similarity_score']:.3f}):\n"
                context += f"Incident: {metadata['IncidentId']}\n"
                context += f"Title: {metadata['AlertTitle']}\n"
                context += f"Category: {metadata['Category']}\n"
                context += f"Severity: {metadata['Severity']}\n"
                context += f"MITRE: {metadata['MitreTechnique']}\n"
                context += f"Description: {metadata['Description']}\n"
                context += f"Systems: {metadata['AffectedSystems']}\n"
                context += f"Remediation: {metadata['RemediationSteps']}\n\n"
            
            # Create prompt for Gemini
            prompt = f"""
You are a cybersecurity expert assistant specializing in Microsoft Security Incident analysis. 
Use the provided security incident context to answer the user's question accurately and professionally.

{context}

User Question: {query}

Instructions:
- Provide accurate, security-focused responses based on the context
- Include specific incident IDs, MITRE techniques, and remediation steps when relevant
- If asking about a specific incident, provide detailed analysis
- Use clear formatting with sections and bullet points
- Always prioritize security best practices
- Be concise but comprehensive

Response:"""

            # Generate response using Gemini
            response = self.gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"⚠️  Gemini API error: {e}")
            return self.generate_fallback_response(query, relevant_docs)
    
    def generate_fallback_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """Generate fallback response without LLM."""
        if not relevant_docs:
            return "❓ No relevant security incidents found for your query. Try asking about incident IDs (INC-2024-001 to INC-2024-005) or security topics like 'phishing', 'ransomware', 'PowerShell', etc."
        
        # Use most relevant document
        top_doc = relevant_docs[0]
        metadata = top_doc['metadata']
        
        return f"""
🔍 **Most Relevant Security Incident** (Similarity: {top_doc['similarity_score']:.3f})

**{metadata['IncidentId']}**: {metadata['AlertTitle']}
- **Category**: {metadata['Category']}
- **Severity**: {metadata['Severity']}
- **MITRE Technique**: {metadata['MitreTechnique']}

**📝 Description:**
{metadata['Description']}

**🖥️ Affected Systems:**
{metadata['AffectedSystems']}

**🔧 Remediation Steps:**
{metadata['RemediationSteps']}

**📊 Status**: {metadata['Status']} | **👥 Assigned**: {metadata['AssignedTo']}
"""
    
    def chat(self, query: str) -> Dict:
        """Main chat interface with RAG."""
        try:
            start_time = datetime.now()
            
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_documents(query, top_k=3)
            
            # Generate response using RAG
            response = self.generate_rag_response(query, relevant_docs)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "query": query,
                "response": response,
                "relevant_documents": len(relevant_docs),
                "top_similarity_score": relevant_docs[0]['similarity_score'] if relevant_docs else 0.0,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "model_used": "gemini-2.0-flash + tfidf" if self.gemini_model else "tfidf-only"
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "response": "❌ Sorry, I encountered an error processing your request.",
                "timestamp": datetime.now().isoformat()
            }

def main():
    """Run the Lightweight RAG Security Chatbot."""
    print("🛡️  Lightweight RAG Security Chatbot")
    print("=" * 60)
    print("Fast startup • TF-IDF similarity • Gemini 2.0 Flash")
    print("Type 'quit' to exit, 'help' for sample queries\n")
    
    try:
        # Initialize chatbot (should be fast)
        chatbot = LightweightRAGChatbot()
        
        # Sample queries for testing
        sample_queries = [
            "Tell me about incident INC-2024-001",
            "What PowerShell security incidents do we have?",
            "Show me all critical severity incidents",
            "How to remediate ransomware attacks?",
            "What is MITRE technique T1566?",
            "List all phishing incidents",
            "What systems were affected by malware?"
        ]
        
        while True:
            user_input = input("🔍 Security Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye! Stay secure! 🛡️")
                break
            
            if user_input.lower() == 'help':
                print("\n💡 Sample Queries to Try:")
                for i, query in enumerate(sample_queries, 1):
                    print(f"{i}. {query}")
                print()
                continue
            
            if not user_input:
                continue
            
            # Get RAG response
            print("\n🤖 Processing...")
            result = chatbot.chat(user_input)
            
            print(f"\n{result['response']}")
            print(f"\n📊 Retrieved {result['relevant_documents']} documents | "
                  f"Similarity: {result['top_similarity_score']:.3f} | "
                  f"Time: {result['processing_time_seconds']:.2f}s | "
                  f"Model: {result['model_used']}\n")
            
    except KeyboardInterrupt:
        print("\nGoodbye! Stay secure! 🛡️")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

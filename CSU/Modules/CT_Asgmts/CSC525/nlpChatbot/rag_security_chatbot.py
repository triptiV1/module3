"""
Lightweight Closed Domain Learning-based RAG Security Chatbot
Uses Gemini 2.0 Flash with vector embeddings for Microsoft Security Incident analysis.
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

# Vector embeddings and similarity search
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Gemini API
import google.generativeai as genai

class RAGSecurityChatbot:
    """
    Learning-based RAG Security Chatbot using vector embeddings and Gemini 2.0 Flash.
    Closed domain system specialized for Microsoft Security Incident analysis.
    """
    
    def __init__(self, dataset_path: str = "SecurityDataset_Sample.csv"):
        """Initialize the RAG Security Chatbot."""
        self.dataset_path = dataset_path
        self.security_data = None
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        self.embedding_model = None
        self.gemini_model = None
        
        # Security domain knowledge
        self.security_intents = [
            "incident_analysis", "threat_detection", "vulnerability_assessment",
            "remediation_guidance", "mitre_techniques", "security_alerts",
            "compliance_check", "risk_assessment", "forensic_analysis", "general_security"
        ]
        
        # MITRE ATT&CK techniques mapping
        self.mitre_techniques = {
            "T1059": "Command and Scripting Interpreter",
            "T1566": "Phishing", 
            "T1486": "Data Encrypted for Impact (Ransomware)",
            "T1078": "Valid Accounts",
            "T1068": "Exploitation for Privilege Escalation",
            "T1041": "Exfiltration Over C2 Channel",
            "T1021": "Remote Services",
            "T1003": "OS Credential Dumping",
            "T1055": "Process Injection",
            "T1090": "Proxy"
        }
        
        print("🛡️  Initializing RAG Security Chatbot...")
        self.setup_gemini()
        self.load_embedding_model()
        self.load_security_data()
        self.build_vector_index()
        print("✅ RAG Security Chatbot ready!")
    
    def setup_gemini(self):
        """Setup Gemini 2.0 Flash API."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("\n🔑 Gemini API Key Required")
                print("Get your API key from: https://aistudio.google.com/app/apikey")
                api_key = getpass.getpass("Enter your Gemini API key: ").strip()
                
                if not api_key:
                    raise ValueError("API key is required")
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini 2.0 Flash configured successfully")
            
        except Exception as e:
            print(f"⚠️  Gemini setup failed: {e}")
            print("Chatbot will work in retrieval-only mode")
            self.gemini_model = None
    
    def load_embedding_model(self):
        """Load sentence transformer model for embeddings."""
        try:
            print("📥 Loading embedding model (all-MiniLM-L6-v2)...")
            # Set environment variables to avoid mutex issues on macOS
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['OMP_NUM_THREADS'] = '1'
            
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            print("Falling back to simple TF-IDF similarity...")
            self.embedding_model = None
    
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
                "RemediationSteps": "1. Isolate affected systems from network 2. Run full antimalware scan 3. Check for persistence mechanisms 4. Review PowerShell execution policies 5. Update endpoint detection rules",
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
                "RemediationSteps": "1. Block malicious sender domains 2. Remove phishing emails from all mailboxes 3. Reset passwords for affected accounts 4. Implement additional email security rules 5. Conduct security awareness training",
                "Status": "In Progress",
                "AssignedTo": "Email Security Team"
            },
            {
                "IncidentId": "INC-2024-003",
                "AlertTitle": "Ransomware Encryption Activity",
                "Category": "Ransomware",
                "Severity": "Critical",
                "MitreTechnique": "T1486",
                "Description": "Ransomware detected encrypting files on network shares. Ransom note left demanding cryptocurrency payment. Attack appears to have originated from compromised VPN account.",
                "AffectedSystems": "File Server FS-01, Network Shares, 200+ encrypted files",
                "RemediationSteps": "1. Immediately isolate affected systems 2. Restore from clean backups 3. Identify initial attack vector 4. Patch vulnerabilities 5. Implement network segmentation 6. Review VPN access logs",
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
                "RemediationSteps": "1. Apply security patches immediately 2. Review user permissions and access rights 3. Monitor for lateral movement 4. Implement least privilege principles 5. Enhance endpoint monitoring",
                "Status": "Resolved",
                "AssignedTo": "Security Operations"
            },
            {
                "IncidentId": "INC-2024-005",
                "AlertTitle": "Suspicious Account Activity",
                "Category": "Account Compromise",
                "Severity": "High",
                "MitreTechnique": "T1078",
                "Description": "User account showing unusual login patterns including access from multiple geographic locations and unusual working hours. Suspicious file access and data exfiltration attempts detected.",
                "AffectedSystems": "Domain Controller, File Servers, User Account: jsmith@company.com",
                "RemediationSteps": "1. Immediately disable compromised account 2. Force password reset 3. Review access logs and data accessed 4. Check for unauthorized privilege changes 5. Implement additional monitoring",
                "Status": "In Progress",
                "AssignedTo": "Identity Security Team"
            },
            {
                "IncidentId": "INC-2024-006",
                "AlertTitle": "Data Exfiltration via C2 Channel",
                "Category": "Data Exfiltration",
                "Severity": "Critical",
                "MitreTechnique": "T1041",
                "Description": "Malware detected establishing command and control channel and exfiltrating sensitive data. Large volumes of data transferred to external IP addresses over encrypted channels.",
                "AffectedSystems": "Database Server DB-PROD-01, Network Infrastructure",
                "RemediationSteps": "1. Block malicious IP addresses 2. Isolate affected database server 3. Analyze exfiltrated data 4. Implement DLP controls 5. Review network monitoring capabilities 6. Notify stakeholders",
                "Status": "In Progress",
                "AssignedTo": "Data Protection Team"
            },
            {
                "IncidentId": "INC-2024-007",
                "AlertTitle": "Remote Desktop Brute Force Attack",
                "Category": "Brute Force",
                "Severity": "Medium",
                "MitreTechnique": "T1021",
                "Description": "Multiple failed RDP login attempts detected from external IP addresses. Attackers attempting to gain access using common username/password combinations.",
                "AffectedSystems": "RDP Gateway, Domain Controllers",
                "RemediationSteps": "1. Block attacking IP addresses 2. Implement account lockout policies 3. Enable multi-factor authentication for RDP 4. Review and strengthen password policies 5. Consider disabling RDP from internet",
                "Status": "Resolved",
                "AssignedTo": "Network Security Team"
            },
            {
                "IncidentId": "INC-2024-008",
                "AlertTitle": "Credential Dumping Activity",
                "Category": "Credential Theft",
                "Severity": "High",
                "MitreTechnique": "T1003",
                "Description": "Suspicious process detected attempting to dump credentials from LSASS memory. Potential use of Mimikatz or similar credential harvesting tools.",
                "AffectedSystems": "WORKSTATION-IT-12, Domain Controller",
                "RemediationSteps": "1. Isolate affected workstation 2. Force password reset for all potentially compromised accounts 3. Enable credential guard 4. Review privileged account usage 5. Implement additional endpoint protection",
                "Status": "Resolved",
                "AssignedTo": "SOC Team Beta"
            }
        ]
        
        self.security_data = pd.DataFrame(sample_data)
        self.security_data.to_csv(self.dataset_path, index=False)
        print(f"✅ Created sample dataset with {len(sample_data)} incidents")
    
    def build_vector_index(self):
        """Build FAISS vector index from security incident documents."""
        try:
            print("🔍 Building vector index from security incidents...")
            
            # Create documents for embedding
            self.documents = []
            for _, incident in self.security_data.iterrows():
                # Combine relevant fields into searchable document
                doc_text = f"""
                Incident: {incident['IncidentId']}
                Title: {incident['AlertTitle']}
                Category: {incident['Category']}
                Severity: {incident['Severity']}
                MITRE Technique: {incident['MitreTechnique']}
                Description: {incident['Description']}
                Affected Systems: {incident['AffectedSystems']}
                Remediation: {incident['RemediationSteps']}
                Status: {incident['Status']}
                """.strip()
                
                self.documents.append({
                    'text': doc_text,
                    'metadata': incident.to_dict()
                })
            
            if self.embedding_model:
                # Generate embeddings with SentenceTransformers
                print("🧠 Generating embeddings...")
                doc_texts = [doc['text'] for doc in self.documents]
                self.embeddings = self.embedding_model.encode(doc_texts, show_progress_bar=False)
                
                # Build FAISS index
                print("📊 Building FAISS index...")
                dimension = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(self.embeddings)
                self.faiss_index.add(self.embeddings.astype('float32'))
                
                print(f"✅ Vector index built with {len(self.documents)} documents")
            else:
                # Fallback to simple text matching
                print("📊 Using simple text similarity (TF-IDF fallback)")
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                doc_texts = [doc['text'] for doc in self.documents]
                self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
                
                print(f"✅ TF-IDF index built with {len(self.documents)} documents")
            
        except Exception as e:
            print(f"❌ Error building vector index: {e}")
            raise
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant documents using vector similarity search."""
        try:
            if self.embedding_model and self.faiss_index:
                # Use SentenceTransformers + FAISS
                query_embedding = self.embedding_model.encode([query])
                faiss.normalize_L2(query_embedding)
                
                # Search in FAISS index
                scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
                
                # Return relevant documents with scores
                relevant_docs = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx != -1:  # Valid index
                        doc = self.documents[idx].copy()
                        doc['similarity_score'] = float(score)
                        doc['rank'] = i + 1
                        relevant_docs.append(doc)
                
                return relevant_docs
            
            else:
                # Fallback to TF-IDF similarity
                query_vector = self.tfidf_vectorizer.transform([query])
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
                context += f"Document {i}:\n"
                context += f"Incident ID: {metadata['IncidentId']}\n"
                context += f"Title: {metadata['AlertTitle']}\n"
                context += f"Category: {metadata['Category']}\n"
                context += f"Severity: {metadata['Severity']}\n"
                context += f"Description: {metadata['Description']}\n"
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
- If the question is about a specific incident, provide detailed analysis
- If asking about general security topics, use the context to provide informed guidance
- Format your response clearly with appropriate sections
- Always prioritize security best practices

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
            return "❓ No relevant security incidents found for your query. Please try asking about specific incident IDs or security topics."
        
        # Use most relevant document
        top_doc = relevant_docs[0]
        metadata = top_doc['metadata']
        
        return f"""
🔍 **Most Relevant Security Incident:**

**{metadata['IncidentId']}**: {metadata['AlertTitle']}
- **Category**: {metadata['Category']}
- **Severity**: {metadata['Severity']}
- **MITRE Technique**: {metadata['MitreTechnique']}

**Description:**
{metadata['Description']}

**Affected Systems:**
{metadata['AffectedSystems']}

**Remediation Steps:**
{metadata['RemediationSteps']}

**Status**: {metadata['Status']}
**Assigned To**: {metadata['AssignedTo']}

💡 *Similarity Score: {top_doc['similarity_score']:.3f}*
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
                "model_used": "gemini-2.0-flash" if self.gemini_model else "retrieval-only"
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "response": "❌ Sorry, I encountered an error processing your request.",
                "timestamp": datetime.now().isoformat()
            }

def main():
    """Run the RAG Security Chatbot."""
    print("🛡️  RAG Security Chatbot - Learning-based Retrieval System")
    print("=" * 70)
    print("Specialized for Microsoft Security Incident analysis using Gemini 2.0 Flash")
    print("Type 'quit' to exit, 'help' for sample queries\n")
    
    try:
        # Initialize chatbot
        chatbot = RAGSecurityChatbot()
        
        # Sample queries for testing
        sample_queries = [
            "Tell me about incident INC-2024-001",
            "What are the PowerShell-related security incidents?",
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
                  f"Processing time: {result['processing_time_seconds']:.2f}s | "
                  f"Model: {result['model_used']}\n")
            
    except KeyboardInterrupt:
        print("\nGoodbye! Stay secure! 🛡️")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

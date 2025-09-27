"""
Simple RAG Security Chatbot - Minimal Dependencies
Uses basic text matching + Gemini 2.0 Flash for Microsoft Security Incident analysis.
"""

import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List
import re
from collections import Counter
import math

# Gemini API
import google.generativeai as genai

class SimpleRAGChatbot:
    """
    Simple RAG Security Chatbot using basic text similarity and Gemini 2.0 Flash.
    Minimal dependencies - guaranteed to work without hanging.
    """
    
    def __init__(self, dataset_path: str = "SecurityDataset_Sample.csv"):
        """Initialize the simple RAG chatbot."""
        self.dataset_path = dataset_path
        self.security_data = None
        self.documents = []
        self.gemini_model = None
        
        print("🛡️  Initializing Simple RAG Security Chatbot...")
        self.setup_gemini()
        self.load_security_data()
        self.build_document_index()
        print("✅ Simple RAG Chatbot ready!")
    
    def setup_gemini(self):
        """Setup Gemini 2.0 Flash API."""
        try:
            # Use the API key directly
            api_key = "AIzaSyDkTO43A946Zxb_60QwjfPDaqHE3MRmMHk"
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini 2.0 Flash configured successfully")
            
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
        """Create expanded sample security incident dataset for testing."""
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
            },
            {
                "IncidentId": "INC-2024-006",
                "AlertTitle": "Data Exfiltration via C2 Channel",
                "Category": "Data Exfiltration",
                "Severity": "Critical",
                "MitreTechnique": "T1041",
                "Description": "Malware detected establishing command and control channel and exfiltrating sensitive customer data. Large volumes of data transferred to external IP addresses over encrypted channels.",
                "AffectedSystems": "Database Server DB-PROD-01, Network Infrastructure",
                "RemediationSteps": "1. Block malicious IP addresses 2. Isolate affected database server 3. Analyze exfiltrated data 4. Implement DLP controls 5. Review network monitoring 6. Notify stakeholders",
                "Status": "In Progress",
                "AssignedTo": "Data Protection Team"
            },
            {
                "IncidentId": "INC-2024-007",
                "AlertTitle": "Remote Desktop Brute Force Attack",
                "Category": "Brute Force",
                "Severity": "Medium",
                "MitreTechnique": "T1021",
                "Description": "Multiple failed RDP login attempts detected from external IP addresses. Attackers attempting to gain access using common username/password combinations and credential stuffing.",
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
                "Description": "Suspicious process detected attempting to dump credentials from LSASS memory. Potential use of Mimikatz or similar credential harvesting tools to extract plaintext passwords.",
                "AffectedSystems": "WORKSTATION-IT-12, Domain Controller",
                "RemediationSteps": "1. Isolate affected workstation 2. Force password reset for all potentially compromised accounts 3. Enable credential guard 4. Review privileged account usage 5. Implement additional endpoint protection",
                "Status": "Resolved",
                "AssignedTo": "SOC Team Beta"
            },
            {
                "IncidentId": "INC-2024-009",
                "AlertTitle": "SQL Injection Attack",
                "Category": "Web Application Attack",
                "Severity": "High",
                "MitreTechnique": "T1190",
                "Description": "SQL injection attack detected on customer portal web application. Attacker attempting to extract database contents and bypass authentication mechanisms.",
                "AffectedSystems": "Web Server WEB-01, Customer Database",
                "RemediationSteps": "1. Take web application offline 2. Patch SQL injection vulnerability 3. Review database access logs 4. Implement input validation 5. Deploy web application firewall",
                "Status": "Resolved",
                "AssignedTo": "Application Security Team"
            },
            {
                "IncidentId": "INC-2024-010",
                "AlertTitle": "Lateral Movement via SMB",
                "Category": "Lateral Movement",
                "Severity": "High",
                "MitreTechnique": "T1021.002",
                "Description": "Compromised account detected moving laterally across network using SMB protocol. Suspicious file transfers and remote command execution observed on multiple systems.",
                "AffectedSystems": "WORKSTATION-FINANCE-03, SERVER-FILE-02, WORKSTATION-HR-07",
                "RemediationSteps": "1. Isolate affected systems 2. Disable compromised account 3. Review SMB logs 4. Implement network segmentation 5. Deploy endpoint detection and response tools",
                "Status": "In Progress",
                "AssignedTo": "Incident Response Team"
            },
            {
                "IncidentId": "INC-2024-011",
                "AlertTitle": "Spear Phishing with Malicious Attachment",
                "Category": "Phishing",
                "Severity": "Critical",
                "MitreTechnique": "T1566.001",
                "Description": "Targeted spear phishing email sent to C-level executives containing malicious Word document with macro-based malware. Document exploits CVE-2022-30190 (Follina).",
                "AffectedSystems": "Executive Email Accounts, WORKSTATION-CEO-01",
                "RemediationSteps": "1. Quarantine malicious emails 2. Isolate affected workstation 3. Disable macros in Office applications 4. Apply security patches 5. Enhanced monitoring for executives",
                "Status": "Resolved",
                "AssignedTo": "Executive Protection Team"
            },
            {
                "IncidentId": "INC-2024-012",
                "AlertTitle": "Cryptocurrency Mining Malware",
                "Category": "Malware",
                "Severity": "Medium",
                "MitreTechnique": "T1496",
                "Description": "Cryptocurrency mining malware detected consuming excessive CPU resources on multiple workstations. Malware likely delivered through compromised software downloads.",
                "AffectedSystems": "WORKSTATION-DEV-05, WORKSTATION-MARKETING-02, WORKSTATION-SALES-08",
                "RemediationSteps": "1. Remove mining malware 2. Block mining pool domains 3. Review software installation policies 4. Implement application whitelisting 5. Monitor CPU usage patterns",
                "Status": "Resolved",
                "AssignedTo": "SOC Team Gamma"
            },
            {
                "IncidentId": "INC-2024-013",
                "AlertTitle": "DNS Tunneling Detection",
                "Category": "Command and Control",
                "Severity": "High",
                "MitreTechnique": "T1071.004",
                "Description": "Suspicious DNS queries detected indicating potential DNS tunneling for command and control communication. Unusual query patterns and subdomain lengths observed.",
                "AffectedSystems": "DNS Servers, Network Infrastructure",
                "RemediationSteps": "1. Block suspicious domains 2. Analyze DNS logs 3. Implement DNS filtering 4. Monitor for data exfiltration 5. Review network security policies",
                "Status": "In Progress",
                "AssignedTo": "Network Security Team"
            },
            {
                "IncidentId": "INC-2024-014",
                "AlertTitle": "USB Device Policy Violation",
                "Category": "Policy Violation",
                "Severity": "Medium",
                "MitreTechnique": "T1091",
                "Description": "Unauthorized USB device connected to secure workstation in finance department. Device contained suspicious files and potential malware payload.",
                "AffectedSystems": "WORKSTATION-FINANCE-01",
                "RemediationSteps": "1. Quarantine USB device 2. Scan workstation for malware 3. Review USB device policies 4. Implement USB port controls 5. Security awareness training",
                "Status": "Resolved",
                "AssignedTo": "Physical Security Team"
            },
            {
                "IncidentId": "INC-2024-015",
                "AlertTitle": "Cloud Storage Data Leak",
                "Category": "Data Leak",
                "Severity": "Critical",
                "MitreTechnique": "T1530",
                "Description": "Misconfigured cloud storage bucket discovered containing sensitive customer data accessible to public internet. Data includes PII and financial information.",
                "AffectedSystems": "AWS S3 Bucket, Cloud Infrastructure",
                "RemediationSteps": "1. Secure cloud storage bucket 2. Review access logs 3. Notify affected customers 4. Implement cloud security policies 5. Conduct security audit",
                "Status": "Resolved",
                "AssignedTo": "Cloud Security Team"
            },
            {
                "IncidentId": "INC-2024-016",
                "AlertTitle": "Insider Threat Activity",
                "Category": "Insider Threat",
                "Severity": "High",
                "MitreTechnique": "T1078.002",
                "Description": "Employee detected accessing sensitive files outside normal job responsibilities and downloading large amounts of data before resignation announcement.",
                "AffectedSystems": "File Servers, HR Database, Employee Workstation",
                "RemediationSteps": "1. Revoke employee access 2. Review data access logs 3. Conduct forensic analysis 4. Implement data loss prevention 5. Legal consultation",
                "Status": "In Progress",
                "AssignedTo": "HR Security Team"
            },
            {
                "IncidentId": "INC-2024-017",
                "AlertTitle": "IoT Device Compromise",
                "Category": "IoT Security",
                "Severity": "Medium",
                "MitreTechnique": "T1200",
                "Description": "Smart building sensors detected communicating with suspicious external servers. Devices appear to be part of botnet and sending telemetry data to unauthorized locations.",
                "AffectedSystems": "Building IoT Sensors, Network Infrastructure",
                "RemediationSteps": "1. Isolate IoT devices 2. Update device firmware 3. Change default passwords 4. Implement IoT network segmentation 5. Monitor IoT traffic",
                "Status": "Resolved",
                "AssignedTo": "IoT Security Team"
            },
            {
                "IncidentId": "INC-2024-018",
                "AlertTitle": "Supply Chain Attack",
                "Category": "Supply Chain",
                "Severity": "Critical",
                "MitreTechnique": "T1195.002",
                "Description": "Third-party software update contained malicious code that compromised multiple systems. Backdoor discovered in legitimate software package from trusted vendor.",
                "AffectedSystems": "Multiple Workstations, Servers running affected software",
                "RemediationSteps": "1. Remove compromised software 2. Restore from clean backups 3. Contact software vendor 4. Review supply chain security 5. Implement software integrity checks",
                "Status": "In Progress",
                "AssignedTo": "Supply Chain Security Team"
            },
            {
                "IncidentId": "INC-2024-019",
                "AlertTitle": "Mobile Device Management Bypass",
                "Category": "Mobile Security",
                "Severity": "Medium",
                "MitreTechnique": "T1444",
                "Description": "Corporate mobile device detected with jailbreak/root access bypassing mobile device management controls. Unauthorized applications installed.",
                "AffectedSystems": "Corporate iPhone, Mobile Infrastructure",
                "RemediationSteps": "1. Wipe and re-enroll device 2. Review MDM policies 3. Implement mobile threat defense 4. User security training 5. Monitor mobile device compliance",
                "Status": "Resolved",
                "AssignedTo": "Mobile Security Team"
            },
            {
                "IncidentId": "INC-2024-020",
                "AlertTitle": "API Security Breach",
                "Category": "API Security",
                "Severity": "High",
                "MitreTechnique": "T1190",
                "Description": "REST API endpoint discovered with authentication bypass vulnerability allowing unauthorized access to customer data. API key validation not properly implemented.",
                "AffectedSystems": "API Gateway, Customer Database",
                "RemediationSteps": "1. Fix API authentication 2. Rotate API keys 3. Review API access logs 4. Implement API rate limiting 5. Deploy API security gateway",
                "Status": "Resolved",
                "AssignedTo": "API Security Team"
            },
            {
                "IncidentId": "INC-2024-021",
                "AlertTitle": "Social Engineering Attack",
                "Category": "Social Engineering",
                "Severity": "High",
                "MitreTechnique": "T1566.004",
                "Description": "Attacker impersonated IT support via phone call to trick employee into installing remote access software and providing credentials.",
                "AffectedSystems": "Employee Workstation, Help Desk Systems",
                "RemediationSteps": "1. Remove remote access software 2. Reset compromised credentials 3. Security awareness training 4. Implement caller verification procedures 5. Monitor for suspicious activity",
                "Status": "Resolved",
                "AssignedTo": "Security Awareness Team"
            },
            {
                "IncidentId": "INC-2024-022",
                "AlertTitle": "Zero-Day Exploit Detection",
                "Category": "Zero-Day",
                "Severity": "Critical",
                "MitreTechnique": "T1068",
                "Description": "Unknown exploit detected targeting previously undiscovered vulnerability in network management software. Exploit provides administrative access to network devices.",
                "AffectedSystems": "Network Management Server, Network Switches",
                "RemediationSteps": "1. Isolate affected systems 2. Apply emergency patches 3. Contact vendor for security update 4. Implement compensating controls 5. Threat intelligence sharing",
                "Status": "In Progress",
                "AssignedTo": "Zero-Day Response Team"
            },
            {
                "IncidentId": "INC-2024-023",
                "AlertTitle": "Container Security Incident",
                "Category": "Container Security",
                "Severity": "High",
                "MitreTechnique": "T1610",
                "Description": "Malicious container image deployed to Kubernetes cluster containing cryptocurrency mining software and network scanning tools.",
                "AffectedSystems": "Kubernetes Cluster, Container Registry",
                "RemediationSteps": "1. Remove malicious containers 2. Scan container images 3. Implement image signing 4. Review container security policies 5. Monitor container runtime",
                "Status": "Resolved",
                "AssignedTo": "Container Security Team"
            },
            {
                "IncidentId": "INC-2024-024",
                "AlertTitle": "Email Account Takeover",
                "Category": "Account Takeover",
                "Severity": "High",
                "MitreTechnique": "T1078.003",
                "Description": "Executive email account compromised through credential stuffing attack. Attacker sent fraudulent wire transfer requests to finance team.",
                "AffectedSystems": "Email Server, Executive Account",
                "RemediationSteps": "1. Secure compromised account 2. Review sent emails 3. Alert finance team 4. Implement email authentication 5. Enable MFA for all executives",
                "Status": "Resolved",
                "AssignedTo": "Email Security Team"
            },
            {
                "IncidentId": "INC-2024-025",
                "AlertTitle": "Network Segmentation Bypass",
                "Category": "Network Security",
                "Severity": "High",
                "MitreTechnique": "T1021.001",
                "Description": "Attacker discovered method to bypass network segmentation controls and access restricted network segments containing sensitive systems.",
                "AffectedSystems": "Network Infrastructure, Segmented Networks",
                "RemediationSteps": "1. Review network segmentation rules 2. Implement micro-segmentation 3. Deploy network access control 4. Monitor inter-segment traffic 5. Update firewall policies",
                "Status": "In Progress",
                "AssignedTo": "Network Architecture Team"
            }
        ]
        
        self.security_data = pd.DataFrame(sample_data)
        self.security_data.to_csv(self.dataset_path, index=False)
        print(f"✅ Created sample dataset with {len(sample_data)} incidents")
    
    def build_document_index(self):
        """Build simple document index from security incidents."""
        try:
            print("🔍 Building document index from security incidents...")
            
            # Create documents for indexing
            self.documents = []
            for _, incident in self.security_data.iterrows():
                # Combine relevant fields into searchable document
                doc_text = f"""
                {incident['IncidentId']} {incident['AlertTitle']} {incident['Category']} 
                {incident['Severity']} {incident['MitreTechnique']} {incident['Description']} 
                {incident['AffectedSystems']} {incident['RemediationSteps']} {incident['Status']}
                """.strip().lower()
                
                self.documents.append({
                    'text': doc_text,
                    'words': set(re.findall(r'\w+', doc_text)),
                    'metadata': incident.to_dict()
                })
            
            print(f"✅ Document index built with {len(self.documents)} documents")
            
        except Exception as e:
            print(f"❌ Error building document index: {e}")
            raise
    
    def calculate_similarity(self, query: str, document: Dict) -> float:
        """Calculate simple word overlap similarity."""
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = document['words']
        
        if not query_words or not doc_words:
            return 0.0
        
        # Jaccard similarity (intersection over union)
        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)
        
        return intersection / union if union > 0 else 0.0
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant documents using simple similarity."""
        try:
            # Calculate similarities
            scored_docs = []
            for doc in self.documents:
                score = self.calculate_similarity(query, doc)
                if score > 0:
                    doc_copy = doc.copy()
                    doc_copy['similarity_score'] = score
                    scored_docs.append(doc_copy)
            
            # Sort by similarity and return top-k
            scored_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Add rank
            for i, doc in enumerate(scored_docs[:top_k]):
                doc['rank'] = i + 1
            
            return scored_docs[:top_k]
            
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
                "model_used": "gemini-2.0-flash + simple-similarity" if self.gemini_model else "simple-similarity-only"
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "response": "❌ Sorry, I encountered an error processing your request.",
                "timestamp": datetime.now().isoformat()
            }

def main():
    """Run the Simple RAG Security Chatbot."""
    print("🛡️  Simple RAG Security Chatbot")
    print("=" * 50)
    print("Fast • Reliable • Gemini 2.0 Flash Powered")
    print("Type 'quit' to exit, 'help' for sample queries\n")
    
    try:
        # Initialize chatbot (should be very fast)
        chatbot = SimpleRAGChatbot()
        
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

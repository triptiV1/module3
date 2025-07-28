import os
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load SpaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Function to load resumes from the current directory
def load_resumes_from_directory():
    current_directory = os.getcwd()  # Get the current working directory
    print(f"Looking for PDF resumes in: {current_directory}")
    
    resumes = []
    for file_name in os.listdir(current_directory):
        if file_name.endswith(".pdf"):  # Look for files with .pdf extension
            file_path = os.path.join(current_directory, file_name)
            print(f"Processing: {file_name}")
            text = extract_text_from_pdf(file_path)
            resumes.append({"File Name": file_name, "Content": text})
    if not resumes:
        raise FileNotFoundError("No PDF files found in the current directory.")
    return pd.DataFrame(resumes)

# Function to preprocess text using SpaCy
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Function to extract skills from resume content
def extract_skills(text):
    """Extract skills from resume content."""
    required_skills = ["Python", "SQL", "NLP", "Java", "Cloud", "AWS Certified", "Azure Certified"]
    found_skills = [skill for skill in required_skills if skill.lower() in text.lower()]
    return found_skills

# Function to extract years of experience from resume content
def extract_years_of_experience(text):
    """Extract years of experience from resume content."""
    import re
    match = re.search(r"(\d+)\s*(\+|\b)\s*years", text.lower())
    if match:
        return int(match.group(1))  # Return the number of years
    return 0  # Default if no experience found

# Function to extract education details (basic example)
def extract_education(text):
    """Extract education from resume content."""
    if "master's" in text.lower():
        return "Master's degree"
    elif "bachelor's" in text.lower():
        return "Bachelor's degree"
    return "None"

# Function to check certifications in resume
def extract_certifications(text):
    """Extract certifications from resume content."""
    certifications = ["AWS Certified", "Azure Certified"]
    found_certifications = [cert for cert in certifications if cert.lower() in text.lower()]
    return found_certifications

# Function to score resumes based on the job description
def score_resumes(resumes, job_description):
    # Preprocess job description and resume content
    job_description = preprocess_text(job_description)
    resumes["Processed_Content"] = resumes["Content"].apply(preprocess_text)

    # Vectorize text using TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([job_description] + resumes["Processed_Content"].tolist())

    # Compute similarity scores
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Extract additional details from resumes
    resumes["Skills"] = resumes["Content"].apply(extract_skills)
    resumes["Skills Match Percentage"] = resumes["Skills"].apply(lambda x: len(x) / len(["Python", "SQL", "NLP"]) * 100)
    resumes["Experience"] = resumes["Content"].apply(extract_years_of_experience)
    resumes["Education"] = resumes["Content"].apply(extract_education)
    resumes["Certifications"] = resumes["Content"].apply(extract_certifications)

    # Add similarity scores to the DataFrame
    resumes["Similarity Score"] = similarity_scores
    resumes = resumes.sort_values(by="Similarity Score", ascending=False)
    return resumes

# Main function to run the program
def main():
    # Load resumes from the same directory as the script
    resumes = load_resumes_from_directory()

    # Load job description
    job_description_file = "job_description.txt"
    print(f"\nLooking for job description file at: {os.path.abspath(job_description_file)}")
    if not os.path.exists(job_description_file):
        raise FileNotFoundError(f"{job_description_file} not found. Please ensure it exists in the same directory.")
    
    with open(job_description_file, "r") as file:
        job_description = file.read()

    # Score resumes and extract details
    ranked_resumes = score_resumes(resumes, job_description)
    
    # Display ranked candidates with detailed output
    print("\nRanked Candidates:")
    for idx, row in ranked_resumes.iterrows():
        print(f"{idx + 1}. {row['File Name']}")
        print(f"   - Similarity Score: {row['Similarity Score']:.2f}")
        print(f"   - Key Skills Match: {row['Skills Match Percentage']:.0f}% (Matched skills: {', '.join(row['Skills'])})")
        print(f"   - Years of Experience: {row['Experience']} (Requirement: 4)")
        print(f"   - Education: {row['Education']} (Meets requirement)")
        certifications = ', '.join(row['Certifications']) if row['Certifications'] else "None"
        cert_status = "Meets requirement" if row['Certifications'] else "Does not meet requirement"
        print(f"   - Certifications: {certifications} ({cert_status})")
        print("-" * 50)

# Run the program
if __name__ == "__main__":
    main()

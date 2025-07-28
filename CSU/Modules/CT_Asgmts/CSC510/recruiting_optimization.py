import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load Spacy's NLP model
nlp = spacy.load("en_core_web_sm")

# List of technical terms and keywords to exclude from name extraction
EXCLUDED_TERMS = {"java", "python", "c++", "machine learning", "vba macros", "sql", "data analysis"}

# Function to extract text from a single PDF file
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text.strip()
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return None  # Return None if the file cannot be read

# Function to check if a string is a valid human name
def is_valid_name(name):
    words = name.split()
    if len(words) < 2 or len(words) > 3:  # Valid names usually have 2-3 words
        return False
    if any(word.lower() in EXCLUDED_TERMS for word in words):  # Exclude technical terms
        return False
    if not all(word[0].isupper() for word in words):  # Ensure proper capitalization
        return False
    return True

# Function to extract a name from the resume text
def extract_name_from_text(text):
    doc = nlp(text)
    # Search for entities labeled as PERSON
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if is_valid_name(name):
                return name
    # Fallback: Check the first few lines of the resume
    lines = text.split("\n")
    for line in lines[:5]:  # Limit to the first 5 lines
        doc_line = nlp(line.strip())
        for ent in doc_line.ents:
            if ent.label_ == "PERSON" and is_valid_name(ent.text.strip()):
                return ent.text.strip()
    return "Unknown"

# Function to process multiple PDF files
def input_resumes_from_pdfs(pdf_files):
    resumes = []
    for file_path in pdf_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        text = extract_text_from_pdf(file_path)
        if text:  # Only process files with valid text
            name = extract_name_from_text(text)
            resumes.append({"name": name, "text": text})
        else:
            print(f"Skipping corrupted or unreadable file: {file_path}")
    return resumes

# Function to preprocess text using NLP
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Function to calculate similarity scores
def calculate_similarity(resumes, job_description):
    texts = [resume['processed_text'] for resume in resumes]
    texts.append(job_description)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    for i, score in enumerate(similarity_scores):
        resumes[i]['similarity'] = score
    return resumes

# Main program
if __name__ == "__main__":
    # Example job description
    job_description = "Looking for a Python Developer with experience in Machine Learning, Data Analysis, and APIs."

    # Automatically detect all PDF files in the current directory
    current_directory = os.getcwd()
    pdf_files = [file for file in os.listdir(current_directory) if file.endswith(".pdf")]

    # Check if any PDF files were found
    if not pdf_files:
        print(f"No PDF files were found in the directory: {current_directory}")
        exit()

    print("Detected PDF Files:", pdf_files)

    # Process resumes
    resumes = input_resumes_from_pdfs(pdf_files)
    if not resumes:
        print("No resumes were processed. Please check the file paths or file contents.")
        exit()

    # Preprocess resumes and calculate similarity
    for resume in resumes:
        resume['processed_text'] = preprocess_text(resume['text'])
    job_description_processed = preprocess_text(job_description)
    resumes_with_scores = calculate_similarity(resumes, job_description_processed)
    ranked_resumes = sorted(resumes_with_scores, key=lambda x: x['similarity'], reverse=True)

    # Output results (Name and Similarity Score only)
    print("\nJob Description:")
    print(job_description)
    print("\nTop Candidates:")
    for candidate in ranked_resumes:
        print(f"Name: {candidate['name']}")
        print(f"Similarity: {round(candidate['similarity'], 2)}")
        print("-" * 50)

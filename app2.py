import streamlit as st
import PyPDF2 as pdf
import spacy
from spacy.cli.download import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import textstat
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# download("en_core_web_sm")

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the soft skills and hard skills lists
soft_skills = ["communication", "teamwork", "leadership", "problem-solving", "critical thinking"]
hard_skills = ["python", "machine learning", "data analysis", "predictive modeling", "data visualization"]

# Function to score the resume
def score_resume(resume_text, job_description=None):
    # Preprocess the resume text
    doc = nlp(resume_text)
    resume_keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    # If job description is provided, preprocess it and calculate the TF-IDF scores and cosine similarity
    if job_description:
        doc = nlp(job_description)
        target_keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([" ".join(target_keywords), " ".join(resume_keywords)])
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1]) + 100
    else:
        similarity_score = 1.0  # If no job description is provided, assume a perfect match

    # Calculate other scores
    impact_score = calculate_impact_score(resume_text)
    brevity_score = calculate_brevity_score(resume_text)
    style_score = calculate_style_score(resume_text)
    sections_score = calculate_sections_score(resume_text)
    soft_skills_score = calculate_soft_skills_score(resume_text, soft_skills)
    hard_skills_score = calculate_hard_skills_score(resume_text, hard_skills)

    # Calculate the overall ATS score
    if job_description:
        overall_ats_score = int((similarity_score * 0.4) + (sum([impact_score, brevity_score, style_score, sections_score]) / 400) * 0.6)
    else:
        overall_ats_score = int((1 * sum([impact_score, brevity_score, style_score, sections_score])) / 400 * 100)
    
    # Suggestions for improvement
    suggestions = generate_suggestions(resume_text, similarity_score, impact_score, brevity_score, style_score, sections_score, soft_skills_score, hard_skills_score)

    # Profile summary
    profile_summary = generate_profile_summary(resume_text)

    # Construct the response
    response = {
        "Overall ATS Score": overall_ats_score,
        "Impact Score": impact_score,
        "Brevity Score": brevity_score,
        "Style Score": style_score,
        "Sections Score": sections_score,
        "Soft Skills Score": soft_skills_score,
        "Hard Skills Score": hard_skills_score,
        "Suggestions": suggestions,
        "Profile Summary": profile_summary
    }

    return response

# Helper functions
def calculate_impact_score(resume_text):
    # Quantifying impact
    impact_pattern = r'\b\d+\%?\b|\$\d+|\d+\s?(million|thousand|billion)'
    impact_matches = re.findall(impact_pattern, resume_text, re.IGNORECASE)
    quantified_impact_score = len(impact_matches) / len(resume_text.split()) * 50 * 20

    # Accomplishment-oriented language
    accomplishment_words = ["achieved", "accomplished", "improved", "increased", "enhanced", "optimized"]
    accomplishment_words_count = sum(resume_text.lower().count(word) for word in accomplishment_words)
    accomplishment_language_score = (accomplishment_words_count / len(resume_text.split())) * 50 * 20

    # Calculate the overall impact score
    impact_score = int(quantified_impact_score + accomplishment_language_score)
    return impact_score

def calculate_brevity_score(resume_text):
    # Resume length
    word_count = len(resume_text.split())
    resume_length_score = 0
    if 450 <= word_count <= 650:
        resume_length_score = 30
    elif 650 < word_count <= 850:
        resume_length_score = 20
    else:
        resume_length_score = 10

    # Use of bullet points
    bullet_points = re.findall(r'^\s*[-*•]\s', resume_text, re.MULTILINE)
    bullet_points_score = len(bullet_points) / len(resume_text.split()) * 30

    # Total bullet points
    total_bullet_points_score = min(len(bullet_points), 20) / 20 * 20

    # Accomplishment-oriented language
    accomplishment_words = ["achieved", "accomplished", "improved", "increased", "enhanced", "optimized"]
    accomplishment_words_count = sum(resume_text.lower().count(word) for word in accomplishment_words)
    accomplishment_language_score = (accomplishment_words_count / len(resume_text.split())) * 20

    # Calculate the overall brevity score
    brevity_score = int(resume_length_score + bullet_points_score + total_bullet_points_score + accomplishment_language_score)
    return brevity_score

def calculate_style_score(resume_text):
    # Check for consistent formatting, headings, and bullet points
    style_score = 80  # Adjust the base score as needed
    if re.search(r'^[A-Z][a-z]+\s*\n=+\s*$', resume_text, re.MULTILINE):
        style_score += 10  # Headings with underlines
    if re.search(r'^\s*[-*•]\s', resume_text, re.MULTILINE):
        style_score += 10  # Bullet points
    return min(100, style_score)

def calculate_sections_score(resume_text):
    # Check for the presence of common resume sections
    sections = ["summary", "experience", "education", "skills", "projects"]
    sections_found = [section for section in sections if re.search(r'\b' + section + r'\b', resume_text, re.IGNORECASE)]
    sections_score = len(sections_found) / len(sections) * 100
    return int(sections_score)

def calculate_soft_skills_score(resume_text, soft_skills):
    # Check for the presence of soft skills
    soft_skills_found = [skill for skill in soft_skills if re.search(r'\b' + skill + r'\b', resume_text, re.IGNORECASE)]
    soft_skills_score = len(soft_skills_found) / len(soft_skills) * 100
    return int(soft_skills_score)

def calculate_hard_skills_score(resume_text, hard_skills):
    # Check for the presence of hard skills
    hard_skills_found = [skill for skill in hard_skills if re.search(r'\b' + skill + r'\b', resume_text, re.IGNORECASE)]
    hard_skills_score = len(hard_skills_found) / len(hard_skills) * 100
    return int(hard_skills_score)

def generate_suggestions(resume_text, similarity_score, impact_score, brevity_score, style_score, sections_score, soft_skills_score, hard_skills_score):
    suggestions = []
    if similarity_score < 0.5:
        suggestions.append("Tailor your resume to better match the job description.")
    if impact_score < 90:
        suggestions.append("Quantify your achievements and showcase your impact more effectively.")
    if brevity_score < 50:
        suggestions.append("Condense your resume and avoid unnecessary details.")
    if style_score < 70:
        suggestions.append("Improve the formatting and structure of your resume for better readability.")
    if sections_score < 80:
        suggestions.append("Ensure your resume includes all the essential sections (Summary, Experience, Education, Skills).")
    if soft_skills_score < 70:
        suggestions.append("Highlight relevant soft skills for the target roles.")
    if hard_skills_score < 70:
        suggestions.append("Emphasize your hard skills and technical expertise relevant to the job.")
    return suggestions

def generate_profile_summary(resume_text):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Generate a 200-300 word technical summary of a candidate's skills, experience, and background based on the following resume text:\n\n{resume_text}"

    response = model.generate_content(prompt)
    profile_summary = response.text

    return profile_summary

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Streamlit app
st.set_page_config(page_title="Smart ATS", page_icon=":briefcase:")
st.title("ATS Scorer")
st.markdown("### Improve Your Resume ATS")

uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the PDF", key="resume_upload")

with st.expander("Add Job Description (Optional)"):
    job_description = st.text_area("Paste the Job Description", height=200, key="job_description")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        text = input_pdf_text(uploaded_file)
        resume_score = score_resume(text, job_description)

        st.success("Resume Analysis Completed!")

        st.markdown("#### Overall ATS Score", help="This is ATS Score")
        st.markdown(f"**{resume_score['Overall ATS Score']}%**")

        st.markdown("#### Impact Score")
        st.markdown(f"**{resume_score['Impact Score']}%**")

        st.markdown("#### Brevity Score")
        st.markdown(f"**{resume_score['Brevity Score']}%**")

        st.markdown("#### Style Score")
        st.markdown(f"**{resume_score['Style Score']}%**")

        st.markdown("#### Sections Score")
        st.markdown(f"**{resume_score['Sections Score']}%**")

        st.markdown("#### Suggestions to Improve ATS Score")
        for suggestion in resume_score['Suggestions']:
            st.markdown(f"- {suggestion}")

        st.markdown("#### Profile Summary")
        st.markdown(resume_score['Profile Summary'])

    else:
        st.warning("Please upload a resume.")

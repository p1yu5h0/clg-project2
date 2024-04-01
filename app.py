import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_repsonse(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Prompt Template
input_prompt = """
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field, software engineering, data science, data analyst
and big data engineer. Your task is to evaluate the resume based on the following categories:

Impact: How well the resume quantifies achievements and showcases the candidate's impact.
Brevity: How concise and focused the resume is, avoiding unnecessary details.
Style: The overall formatting, structure, and readability of the resume.
Sections: The completeness and organization of the resume sections (e.g., Summary, Experience, Education, Skills).
Soft Skills: The presence and emphasis on relevant soft skills for the target roles.
Hard Skills: The inclusion and prominence of relevant hard skills for the target roles.
Suggestions: This include the suggestions by the ATS for inproving the resume score and help the student to secure a job easily. Tell about 5-6 things by which you could improve the resume.

{jd_prompt}
resume:{text}

I want the response in one single string having the structure
{{"Overall ATS Score":"%", "Impact Score":"%", "Brevity Score":"%", "Style Score":"%", "Sections Score":"%", "Soft Skills Score":"%", "Hard Skills Score":"%", "MissingKeywords":[], "Profile Summary":"", "Suggestions": ""}}
"""

# Streamlit app
st.set_page_config(page_title="Smart ATS", page_icon=":briefcase:")
st.title("ATS Scorer")
st.markdown("### Improve Your Resume ATS")

uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the PDF", key="resume_upload")

with st.expander("Add Job Description (Optional)"):
    jd = st.text_area("Paste the Job Description", height=200, key="job_description")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        text = input_pdf_text(uploaded_file)
        if jd:
            jd_prompt = f"You must consider the job market is very competitive and you should provide best assistance for improving the resumes. Assign scores based on the given job description:\n{jd}"
        else:
            jd_prompt = "You must provide scores for the resume based on the categories mentioned above."

        response = get_gemini_repsonse(input_prompt.format(text=text, jd_prompt=jd_prompt))
        try:
            response_data = json.loads(response)
            st.success("Resume Analysis Completed!")

            st.markdown("#### Overall ATS Score")
            st.markdown(f"**{response_data['Overall ATS Score']}**")

            st.markdown("#### Impact Score")
            st.markdown(f"**{response_data['Impact Score']}**")

            st.markdown("#### Brevity Score")
            st.markdown(f"**{response_data['Brevity Score']}**")

            st.markdown("#### Style Score")
            st.markdown(f"**{response_data['Style Score']}**")

            st.markdown("#### Sections Score")
            st.markdown(f"**{response_data['Sections Score']}**")

            st.markdown("#### Soft Skills Score")
            st.markdown(f"**{response_data['Soft Skills Score']}**")

            st.markdown("#### Hard Skills Score")
            st.markdown(f"**{response_data['Hard Skills Score']}**")

            st.markdown("#### Missing Keywords")
            if response_data['MissingKeywords']:
                for keyword in response_data['MissingKeywords']:
                    st.markdown(f"- {keyword}")
            else:
                st.markdown("No missing keywords found.")

            st.markdown("#### Profile Summary")
            st.markdown(response_data['Profile Summary'])

            st.markdown("#### Suggestions to Improve ATS Score")
            st.markdown(response_data['Suggestions'])


        except (ValueError, KeyError):
            st.error("Invalid response format. Please try again.")
    else:
        st.warning("Please upload a resume.")

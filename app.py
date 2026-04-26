
import streamlit as st
import json
import spacy
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import fitz  
import io
import pickle
import re

# Add your key here
MY_GROQ_API_KEY = "PASTE_YOUR_API_KEY_HERE"

st.set_page_config(layout="wide", page_title="Smart ATS Project")

# --- Load Models Once ---
@st.cache_resource
def load_spacy_model():
    return spacy.load('en_core_web_sm')

@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('all-mpnet-base-v2')

@st.cache_resource
def load_rf_model():
    # Loading YOUR custom trained machine learning model!
    with open('custom_scoring_model.pkl', 'rb') as f:
        return pickle.load(f)

nlp = load_spacy_model()
embedding_model = load_sentence_transformer_model()
custom_rf_model = load_rf_model()

# --- Functions (Copied from previous steps for self-contained app) ---
def redact_resume(text):
    doc = nlp(text)
    redacted_text = list(text)

    # Words we do NOT want to redact, even if spaCy thinks they are an ORG or PERSON
    ignore_list = ['air quality', 'data analysis', 'excel', 'python', 'power bi', 'zoom', 'canva', 'research']

    for ent in sorted(doc.ents, key=lambda x: x.start_char, reverse=True):
        # Skip redaction if the entity text (lowercased) is in our ignore list
        if ent.text.lower() in ignore_list:
            continue

        if ent.label_ == 'PERSON':
            placeholder = '[PERSON]'
        elif ent.label_ == 'DATE':
            # Only redact years or specific dates to protect age, but leave general timeframes alone
            placeholder = '[DATE]'
        elif ent.label_ == 'ORG':
            placeholder = '[ORG]'
        else:
            continue

        redacted_text[ent.start_char:ent.end_char] = list(placeholder)

    return "".join(redacted_text)

def extract_resume_data(redacted_text):
    if not MY_GROQ_API_KEY or MY_GROQ_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        return {"error": "Groq API Key is required for data extraction.", "llm_output": ""}

    try:
        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key=MY_GROQ_API_KEY)

        prompt = PromptTemplate.from_template(
            """
            You are an expert HR extraction system. Read the following resume text and
            extract the data into a strict JSON format with the exact following keys:

            - "skills": [Extract ALL technical tools, software, AND domain skills (e.g., 'Data Analysis', 'Reporting', 'Research', 'Teamwork', 'Project Management') mentioned anywhere in the text. Be exhaustive and include both hard and soft skills.]
            - "education_level": "Highest degree obtained (e.g., B.Sc., M.Sc., High School)"
            - "years_of_experience": Calculate the total years of professional working experience as an integer. If the experience is purely academic/thesis work, output 0.

            Resume Text: {text}

            Return ONLY valid JSON. Do not include any markdown formatting like ```json, and do not include any conversational text.
            """
        )

        chain = prompt | llm
        response = chain.invoke({"text": redacted_text})

        # Clean up the response in case the LLM still adds markdown
        clean_json_string = response.content.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json_string)

    except Exception as e:
        return {"error": str(e), "llm_output": response.content if 'response' in locals() else "Failed to get response"}

def calculate_match_score(candidate_skills_list, job_description_text):
    if not candidate_skills_list or not job_description_text:
        return 0.0

    resume_text = " ".join(candidate_skills_list)

    # --- Feature 1: Semantic Distance ---
    resume_embedding = embedding_model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = embedding_model.encode(job_description_text, convert_to_tensor=True)
    raw_cosine_score = util.cos_sim(resume_embedding, jd_embedding).item()

    # --- Feature 2: Exact Word Match ---
    cand_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    job_words = set(re.findall(r'\b\w+\b', job_description_text.lower()))

    if len(job_words) > 0:
        exact_match_score = len(cand_words & job_words) / len(job_words)
    else:
        exact_match_score = 0.0

    # --- Final Prediction ---
    predicted_hr_score = custom_rf_model.predict([[raw_cosine_score, exact_match_score]])[0]

    # Ensure score stays between 0 and 100
    final_percentage = max(0.0, min(100.0, predicted_hr_score * 100))
    return round(final_percentage, 2)

def generate_feedback(candidate_skills_list, job_description_text):
    if not MY_GROQ_API_KEY or MY_GROQ_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        return "Groq API Key is required for feedback generation."
        
    llm = ChatGroq(temperature=0.3, groq_api_key=MY_GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    prompt_template = PromptTemplate.from_template(
        """You are an experienced HR advisor. A candidate with the following skills: {candidate_skills} is applying for a role with the following job description: {job_description}.
        Please provide a short, 3-sentence summary of what core skills the candidate appears to be missing based on the job description. Focus only on the skills.
        """
    )
    chain = prompt_template | llm
    response = chain.invoke({
        "candidate_skills": ", ".join(candidate_skills_list),
        "job_description": job_description_text
    })
    return response.content

def extract_text_from_file(uploaded_file):
    """Extracts text from PDF, DOCX, or TXT files."""
    if uploaded_file.name.endswith('.pdf'):
       
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.name.endswith('.txt'):
        return str(uploaded_file.read(), 'utf-8')
    else:
        return ""


st.title("Smart & Unbiased Applicant Tracking System")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Job Description")
    # Recruiters usually copy-paste JDs, so we keep this as a text area
    job_description_text = st.text_area("Paste the Job Description here:", height=200)

with col2:
    st.header("2. Candidate Resume")
    
    uploaded_resume = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'])


if st.button("Analyze Resume"):
    if not job_description_text or not uploaded_resume:
        st.error("Please provide both a Job Description and upload a Resume.")
    else:
        with st.spinner("Analyzing document and extracting data with AI..."):
            
            # 1. Read the uploaded file
            candidate_raw_resume_text = extract_text_from_file(uploaded_resume)
            
            # 2. FIX: Actually create the redacted resume by calling the function!
            redacted_resume = redact_resume(candidate_raw_resume_text)

            st.subheader("1. Candidate Processing")

         
            with st.expander("🕶️ View Redacted Resume (Blind Hiring Mode)"):
                st.info("To prevent unconscious bias, the candidate's personal info, university names, and dates have been hidden before AI evaluation.")
                st.write(redacted_resume)

            st.subheader("2. Extracted Candidate Data (JSON)")
            # 3. Pass the fixed variable to the JSON extractor
            extracted_data = extract_resume_data(redacted_resume)

            if "error" in extracted_data:
                st.error(f"Error during data extraction: {extracted_data['error']}")
                extracted_skills = []
            else:
                st.json(extracted_data)
                extracted_skills = extracted_data.get('skills', [])

            st.subheader("3. Match Score")
            match_score = calculate_match_score(extracted_skills, job_description_text)
            st.metric(label="Skill Match Percentage", value=f"{match_score}%")

            st.subheader("4. LLM Feedback on Missing Skills")
            feedback = generate_feedback(extracted_skills, job_description_text)
            st.info(feedback)
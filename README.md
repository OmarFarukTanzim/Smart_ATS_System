# Smart & Unbiased Applicant Tracking System (ATS)

An AI-powered, unbiased resume evaluation tool. This project goes beyond simple keyword matching by combining **Natural Language Processing (NLP)**, **Large Language Models (LLMs)**, and a **Custom-Tuned Machine Learning Model (XGBoost)** to evaluate candidates fairly and accurately.

## ✨ Key Features

* 🕶️ **Blind Hiring Mode:** Uses `spaCy` to automatically redact Personally Identifiable Information (PII) such as names, universities, and dates to prevent unconscious bias.
* 🧠 **Intelligent Skill Extraction:** Leverages **Groq (Llama 3)** to accurately parse resumes and extract hard/soft skills and years of experience into a strict JSON format.
* 📊 **Advanced Match Scoring:** Calculates a highly realistic match score using a custom-trained **XGBoost Regressor**. The model evaluates both semantic similarity (using HuggingFace's `all-mpnet-base-v2`) and exact keyword overlaps.
* 📄 **Robust Document Parsing:** Uses `PyMuPDF` (`fitz`) and `python-docx` to flawlessly extract text from complex, modern PDF and DOCX formats.
* 💬 **Actionable AI Feedback:** Generates a concise, customized summary highlighting the core skills a candidate is missing based on the job description.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Machine Learning:** XGBoost, Scikit-Learn
* **NLP & Embeddings:** spaCy, HuggingFace Sentence Transformers
* **LLM API:** Groq (Llama-3.1-8b)
* **Document Processing:** PyMuPDF, PyPDF2, python-docx, Pandas

## 📂 File Structure

* `app.py`: The main Streamlit web application script.
* `train_model.py`: The Machine Learning Operations (MLOps) script used to engineer features, run Grid Search hyperparameter tuning, and train the XGBoost model.
* `evaluate_accurcy.py`: Script for evaluating the ML model's performance and error rates.
* `custom_scoring_model.pkl`: The serialized, production-ready XGBoost model.
* `ats.ipynb`: Jupyter Notebook used for initial exploratory data analysis and prototyping.
* `requirements.txt`: List of dependencies required to run the project.

## How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/smart-ats-evaluator.git](https://github.com/YOUR_USERNAME/smart-ats-evaluator.git)
cd smart-ats-evaluator

2. Install dependencies
pip install -r requirements.txt

3. Set up your API Key
Open app.py and replace the placeholder MY_GROQ_API_KEY with your actual Groq API Key.

4. Run the Streamlit app
streamlit run app.py
Model Training Details
The scoring engine is not a simple rule-based system. It is an XGBoost Regressor trained on a dataset of over 7,000 resumes.

Feature 1: Semantic Vector Distance (Cosine Similarity via all-mpnet-base-v2).

Feature 2: Exact Keyword Overlap Percentage.

Optimization: The model underwent Grid Search Hyperparameter Tuning, resulting in an optimized Mean Absolute Error (MAE) of ~12.5%, providing production-grade, highly reliable HR scoring.

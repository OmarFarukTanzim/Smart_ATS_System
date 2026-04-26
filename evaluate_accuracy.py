import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_absolute_error

print("1. Loading the AI Embedding Model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("2. Loading the Ground Truth Dataset...")
# Load the dataset and drop any empty rows
df = pd.read_csv('resume_data.csv')
df = df.dropna(subset=['skills', 'skills_required', 'matched_score'])

# For testing, let's just evaluate the first 100 rows so it runs quickly
test_df = df.head(100).copy()

print(f"3. Evaluating {len(test_df)} resumes. This will take a few seconds...\n")

predicted_scores = []

# Loop through each resume in our test batch
for index, row in test_df.iterrows():
    # 1. Get the candidate's skills and the required skills
    candidate_skills = str(row['skills'])
    job_requirements = str(row['skills_required'])

    candidate_embedding = embedding_model.encode(candidate_skills, convert_to_tensor=True)
    job_embedding = embedding_model.encode(job_requirements, convert_to_tensor=True)
    
    # 3. Calculate the AI Match Score (Cosine Similarity)
    ai_score = util.cos_sim(candidate_embedding, job_embedding).item()
    
    # Save the AI's guess
    predicted_scores.append(ai_score)

# Add the AI's guesses to our dataframe so we can compare them side-by-side
test_df['ai_predicted_score'] = predicted_scores

# Calculate the Mean Absolute Error (MAE)
true_scores = test_df['matched_score'].tolist()
mae = mean_absolute_error(true_scores, predicted_scores)

print("=== EVALUATION RESULTS ===")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"On average, the AI is off by: {mae * 100:.2f}%\n")

# Let's peek at the first 5 to see how close it was visually!
print("Look at the first 5 predictions:")
print(test_df[['matched_score', 'ai_predicted_score']].head(5))
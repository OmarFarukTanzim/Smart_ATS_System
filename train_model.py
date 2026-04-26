import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pickle
import re

print("1. Loading the upgraded AI Embedding Model (all-mpnet-base-v2)...")
embedding_model = SentenceTransformer('all-mpnet-base-v2')

print("2. Loading the Dataset...")
df = pd.read_csv('resume_data.csv').dropna(subset=['skills', 'skills_required', 'matched_score'])


df = df.copy()

print(f"3. Generating Features for {len(df)} resumes (Semantic Similarity & Exact Matches)...")
print("(This takes a moment...)")
raw_scores = []
exact_match_percentages = []

for index, row in df.iterrows():
    cand_str = str(row['skills'])
    job_str = str(row['skills_required'])
    
    # --- Feature 1: Semantic Distance (Cosine Similarity) ---
    cand_emb = embedding_model.encode(cand_str, convert_to_tensor=True)
    job_emb = embedding_model.encode(job_str, convert_to_tensor=True)
    raw_scores.append(util.cos_sim(cand_emb, job_emb).item())
    
    # --- Feature 2: Exact Word Match Percentage ---
    cand_words = set(re.findall(r'\b\w+\b', cand_str.lower()))
    job_words = set(re.findall(r'\b\w+\b', job_str.lower()))
    
    if len(job_words) > 0:
        overlap = len(cand_words & job_words)
        match_percentage = overlap / len(job_words)
    else:
        match_percentage = 0.0
        
    exact_match_percentages.append(match_percentage)

# Add both features to our dataframe
df['raw_ai_score'] = raw_scores
df['exact_match_score'] = exact_match_percentages

print("\n4. Tuning the XGBoost 'Translator' Model...")
print("Executing Grid Search across 27 different setting combinations.")

X = df[['raw_ai_score', 'exact_match_score']]
y = df['matched_score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the base model
base_model = XGBRegressor(random_state=42)

# Define the "Grid" of settings we want to test
param_grid = {
    'n_estimators': [100, 200, 300],       # How many trees to build
    'learning_rate': [0.01, 0.05, 0.1],    # How fast the model learns
    'max_depth': [3, 5, 7]                 # How complex each tree is allowed to get
}

# Run the Grid Search
# cv=3 means it double-checks its work 3 times for every combination
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, 
                           cv=3, scoring='neg_mean_absolute_error', verbose=1)

grid_search.fit(X_train, y_train)

# Get the absolute best model from the search
best_model = grid_search.best_estimator_

# 5. Evaluate the NEW error rate
predictions = best_model.predict(X_test)
new_mae = mean_absolute_error(y_test, predictions)

print("\n=== FINAL TUNING RESULTS ===")
print(f"Best Settings Found: {grid_search.best_params_}")
print(f"NEW Tuned XGBoost Error Rate: {new_mae * 100:.2f}%")

print("\n6. Saving the Best Model...")
with open('custom_scoring_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Saved as 'custom_scoring_model.pkl'!")
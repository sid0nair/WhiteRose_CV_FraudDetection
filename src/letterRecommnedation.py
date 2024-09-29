# Import necessary libraries
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy's large English model with word vectors
nlp = spacy.load('en_core_web_lg')  # Use 'en_core_web_lg' for word vectors

# Load the dataset containing the recommendation phrases
recommendation_df = pd.read_csv('/kaggle/input/resume-and-recommendation/final_recommendation.csv')

# Create a list of reference sentences with exaggerated tones
reference_sentences = [
    "This individual is the best I have ever seen.",
    "Their performance is absolutely outstanding and unmatched.",
    "One of the most remarkable and talented people I have met.",
    "Truly exceptional with unparalleled skills.",
    "An incredible and phenomenal worker who exceeds all expectations."
]

# Convert reference sentences into embeddings
reference_embeddings = [nlp(sentence).vector for sentence in reference_sentences]

# Define a function to calculate the superlative score using cosine similarity
def calculate_superlative_score(text):
    if pd.isna(text) or text.strip() == '':  # Check if the text is NaN or empty
        return 0  # No text, return score 0
    
    # Compute the embedding for the input text
    phrase_embedding = nlp(text).vector

    # Calculate cosine similarity with each reference sentence
    similarities = [cosine_similarity([phrase_embedding], [ref_emb])[0][0] for ref_emb in reference_embeddings]
    
    # Return the maximum similarity as the score
    return max(similarities)

# Combine 'Skills Vouched for' and 'Phrases' columns into a single text column
recommendation_df['Combined Text'] = recommendation_df['Skills Vouched for'].fillna('') + ' ' + recommendation_df['Phrases'].fillna('')

# Apply the function to calculate the superlative scores using the combined text
recommendation_df['Superlative Score'] = recommendation_df['Combined Text'].apply(calculate_superlative_score)

# Step 1: Aggregate the scores for each interviewee (using the mean here, but you can choose median, max, etc.)
aggregated_scores = recommendation_df.groupby('Interviewee ID')['Superlative Score'].mean().reset_index()

# Step 2: Normalize these aggregated scores across all interviewees
min_score = aggregated_scores['Superlative Score'].min()
max_score = aggregated_scores['Superlative Score'].max() + 1e-5  # Avoid division by zero
aggregated_scores['Normalized Superlative Score'] = (aggregated_scores['Superlative Score'] - min_score) / (max_score - min_score)

# Step 3: Merge the normalized score back to the original dataframe
recommendation_df = recommendation_df.merge(aggregated_scores[['Interviewee ID', 'Normalized Superlative Score']], on='Interviewee ID')

# Save the updated DataFrame to a CSV file to facilitate download and inspection on Kaggle
recommendation_df.to_csv('/kaggle/working/updated_recommendation_scores.csv', index=False)

# Display the message confirming the file has been saved
print("The updated DataFrame has been saved as 'updated_recommendation_scores.csv' for inspection.")

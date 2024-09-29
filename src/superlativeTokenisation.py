# Aggregate all job titles from recommendation letters for each interviewee
aggregated_recommendations_df = recommendations_df.groupby('Interviewee ID')['Job Title applying to'].apply(lambda x: ' '.join(x.dropna())).reset_index()

# Merge the aggregated job titles with the resumes data
merged_df = pd.merge(resumes_df_subset, aggregated_recommendations_df, left_on='pdf_id', right_on='Interviewee ID', how='inner')

# Simplified tokenization for aggregated job titles
merged_df['Job Title Tokens'] = merged_df['Job Title'].apply(lambda x: x.split())
merged_df['Aggregated Applying Job Title Tokens'] = merged_df['Job Title applying to'].apply(lambda x: x.split())

# Combine all tokens to train Word2Vec model
all_tokens = merged_df['Job Title Tokens'].tolist() + merged_df['Aggregated Applying Job Title Tokens'].tolist()

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=all_tokens, vector_size=100, window=5, min_count=1, workers=4)

# Calculate similarity between job titles using word embeddings
similarities = []
for index, row in merged_df.iterrows():
    job_title_vec = avg_word_vector(row['Job Title Tokens'], word2vec_model, 100)
    applying_job_title_vec = avg_word_vector(row['Aggregated Applying Job Title Tokens'], word2vec_model, 100)
    
    # Calculate cosine similarity, and handle the case where vectors are zero
    if not np.any(job_title_vec) or not np.any(applying_job_title_vec):
        similarity = 0  # Assign zero similarity if any vector is all zeros
    else:
        similarity = 1 - cosine(job_title_vec, applying_job_title_vec)
    
    similarities.append(similarity)

# Convert similarity to risk factor (inversely proportional)
merged_df['Job Title Risk Factor'] = 1 - np.array(similarities)

# Normalize risk scores to be between 0 and 1
merged_df['Job Title Risk Factor'] = (merged_df['Job Title Risk Factor'] - merged_df['Job Title Risk Factor'].min()) / (merged_df['Job Title Risk Factor'].max() - merged_df['Job Title Risk Factor'].min())

# Display the updated dataframe with improved risk factors
tools.display_dataframe_to_user(name="Aggregated Job Title Risk Factor Analysis", dataframe=merged_df)

merged_df.head()


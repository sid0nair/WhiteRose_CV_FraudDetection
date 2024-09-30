# CV Fraud Detection and Analysis

This directory contains the core scripts and notebooks used for fraud detection, skill ranking, and data analysis for interviewee profiles. The scripts leverage machine learning, natural language processing (NLP), and network analysis to identify fraudulent profiles, rank skills, and perform detailed data analysis.

## Table of Contents

- [Files Overview](#files-overview)
- [Requirements](#requirements)
- [Usage Instructions](#usage-instructions)

## Files Overview

1. **`BiLSTM.ipynb`**: Jupyter notebook for training and testing the BiLSTM autoencoder used for fraud detection.
2. **`BiLSTM_train_model.py`**: Python script for training the BiLSTM model. Designed to be run in environments where Jupyter is not available.
3. **`LORtoCSV.py`**: Converts text files containing Letters of Recommendation (LOR) into structured CSV format, extracting key information for further analysis.
4. **`data_analysis.ipynb`**: Jupyter notebook for performing comprehensive data analysis, including network analysis, sentiment analysis, and skill clustering.
5. **`letterRecommendation.py`**: Analyzes recommendation letters, performs text processing, and extracts features such as superlative phrases and skill mentions.
6. **`superlative.ipynb`**: Jupyter notebook focusing on the analysis of superlative phrases in recommendation letters, providing insights into potential exaggerations.
7. **`readme.md`**: This file provides an overview and usage instructions for the scripts and notebooks in this directory.

## Requirements

- **Python** 3.6 or later
- **Libraries**:
  
  To install all the necessary libraries, use the following commands:
   ```bash
  pip install pandas numpy scikit-learn tensorflow matplotlib networkx textblob spacy seaborn python-louvain
  python -m spacy download en_core_web_sm

## Usage Instructions

### 1. **LOR to CSV Conversion**
   - This step extracts key information from Letters of Recommendation (LORs) and converts them into a structured CSV format.
   - **Script:** `LORtoCSV.py`
   - **Usage:** 
     1. Place the text files containing LORs in the designated input directory (specified within the script).
     2. Open a terminal, navigate to the `src` directory, and run the following command:
        ```bash
        python LORtoCSV.py
        ```
     3. The script will generate a CSV file (`Recommendation_Letter_CSV.csv`) containing extracted details like recommender ID, interviewee ID, job titles, and skills vouched for.

### 2. **Train the BiLSTM Model for Fraud Detection**
   - This step involves training a BiLSTM autoencoder to detect anomalies in the interviewee profiles.
   - **Script:** `BiLSTM_train_model.py`
   - **Usage:** 
     1. Ensure that the combined data file (e.g., `combined_data.csv`) is in the appropriate directory.
     2. Modify the script to set the correct file paths if needed.
     3. In the terminal, navigate to the `src` directory and run:
        ```bash
        python BiLSTM_train_model.py
        ```
     4. The script will train the model using the input data and save the anomaly detection results to a CSV file (`anomaly_detection_output.csv`).

### 3. **Data Analysis and Network Visualization**
   - This step performs a comprehensive data analysis, including network visualization, skill clustering, sentiment analysis, and centrality measurement.
   - **Notebook:** `data_analysis.ipynb`
   - **Usage:** 
     1. Open the notebook in Jupyter Notebook or Google Colab.
     2. Update file paths in the notebook to match the location of your input data (e.g., `combined_data.csv`, `updated_recommendation_scores.csv`).
     3. Run each cell sequentially to:
        - Build a network of interviewees and recommenders.
        - Perform sentiment analysis on the "Skills Vouched for" in recommendation letters.
        - Identify central nodes in the network using degree, closeness, and betweenness centrality.
        - Cluster skillsets and visualize cross-cluster interactions.
     4. Visualizations such as network graphs, centrality plots, and cross-cluster heatmaps will be displayed within the notebook.
     5. Results and metrics will be printed and optionally saved for further analysis.

### 4. **Superlative Phrase Analysis**
   - This step analyzes the use of superlative phrases in recommendation letters to flag potential exaggerations.
   - **Notebook:** `superlative.ipynb`
   - **Usage:** 
     1. Open the notebook in Jupyter Notebook or Google Colab.
     2. Ensure that the recommendation letters data (`Recommendation_Letter_CSV.csv`) is accessible.
     3. Run the cells to:
        - Extract superlative phrases from the recommendation letters.
        - Visualize the frequency and context of superlative usage.
     4. Analyze the output to identify profiles with potentially exaggerated claims based on the presence of superlative phrases.

### 5. **Recommendation Letter Analysis**
   - This script processes the recommendation letters to extract features such as skills and job titles.
   - **Script:** `letterRecommendation.py`
   - **Usage:** 
     1. Ensure that the input file (`Recommendation_Letter_CSV.csv`) is in the correct directory.
     2. In the terminal, navigate to the `src` directory and run:
        ```bash
        python letterRecommendation.py
        ```
     3. The script will process the letters, extract key features, and save the processed data for further analysis.

### Additional Notes

- **Updating File Paths:** Some scripts and notebooks require updating file paths to match the location of your data files. Open the script or notebook and modify the `file_path` variables as needed.
- **Environment:** It is recommended to use a virtual environment for this project to manage dependencies. Run the following commands to create and activate a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate

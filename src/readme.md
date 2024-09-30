# CV Fraud Detection and Analysis

This directory contains the core scripts and notebooks used for fraud detection, skill ranking, and data analysis for interviewee profiles. The scripts leverage machine learning, natural language processing (NLP), and network analysis to identify fraudulent profiles, rank skills, and perform detailed data analysis.

## Table of Contents

- [Files Overview](#files-overview)
- [Requirements](#requirements)
- [Usage Instructions](#usage-instructions)
- [Script Details](#script-details)
- [Outputs](#outputs)
- [Notes](#notes)
- [Future Enhancements](#future-enhancements)

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


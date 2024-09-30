# WhiteRose_Innov8_2.0_Final

# CV Fraud Detection and Skill Ranking System

This project combines fraud detection and skill ranking for profiles using a single input CSV file. It performs the following tasks:

- **Fraud Detection:** Identifies fraudulent profiles based on risk factors using a BiLSTM Autoencoder.
- **Skill Extraction and Ranking:** Extracts, normalizes, and ranks skills from profile data. Fraudulent profiles have their skill scores set to zero.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Input Data](#input-data)
- [Instructions](#instructions)
- [Run the Script](#run-the-script)
- [Outputs](#outputs)
- [Notes](#notes)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contact](#contact)

## Requirements

- **Python** 3.6 or later
- **Libraries:**

  To install the required libraries, run:
  ```bash
  pip install pandas numpy scikit-learn tensorflow matplotlib networkx textblob spacy seaborn python-louvain
  python -m spacy download en_core_web_sm


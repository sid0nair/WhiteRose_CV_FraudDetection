# WhiteRose_Innov8Final

# Fraud Detection and Skill Ranking System

## Overview

This project combines fraud detection and skill ranking for profiles using a single input CSV file. It performs:

- **Fraud Detection**: Identifies fraudulent profiles based on risk factors using a BiLSTM Autoencoder.
- **Skill Extraction and Ranking**: Extracts and normalizes skill counts from profile data. Fraudulent profiles have their skill scores set to zero.

## Requirements

- **Python** 3.6 or later
- **Libraries**:

  ```bash
  pip install pandas numpy scikit-learn tensorflow matplotlib spacy
  python -m spacy download en_core_web_sm
  ```
  ## Input Data

**Combined Data File** (`combined_data.csv`):  
Contains all necessary columns:

- `id`
- `superlative_score`
- `text_inconsistency_score`
- `reciprocity_score`
- `Work_Experience`
- `Education`
- `Skills`

**Recommendation Connections File** (`Reciprocal_Recommendations.csv`):  
Contains recommendation connections with columns:

- `Person_A`
- `Person_B`

## Instructions

**Place Input Files:**  
Ensure `combined_data.csv` and `Reciprocal_Recommendations.csv` are in accessible directories.

**Update File Paths in Script:**  
Modify the script (`fraud_detection_and_skill_ranking.py`) to point to your input files:

```python
data_file_path = '/path/to/combined_data.csv'
connections_data = pd.read_csv('/path/to/Reciprocal_Recommendations.csv')
```
## Run the Script:

```bash
python fraud_detection_and_skill_ranking.py
```

## Outputs:
- fraudulent_profiles.csv: Contains IDs and reasons for profiles flagged as fraudulent.
- normalized_skill_data.csv: Contains IDs and normalized skill counts, sorted in descending order.

## Notes

- Ensure all required libraries are installed and the spaCy model is downloaded.
- The script may take time to execute due to model training and data processing.

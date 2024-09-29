import os
import csv
from transformers import pipeline

# Since the script is run from "EIGHTFOLD_2", use only the subdirectory name
sub_directory = "Final_Recommendation_Letters"
dir_path = os.path.abspath(sub_directory)  # Get the absolute path
print(f"Checking main directory path: {dir_path}")

# Load a pre-trained NER model
print("Loading pre-trained NER model...")
nlp = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
print("Model loaded successfully.")

# Updated keywords to identify certain information
keywords = {
    "experience": [
        "years of experience", "extensive experience", "skilled in", "proficiency", 
        "dedicated", "remarkable expertise", "successfully navigated"
    ],
    "skills": [
        "analytical skills", "communication", "problem-solving", "mentoring", "coaching", 
        "conflict resolution", "benefits administration", "crisis intervention", "advocacy",
        "accounting software", "technical expertise"
    ],
    "achievements": [
        "spearheaded initiatives", "process improvements", "exceeding goals", 
        "successful development", "innovative approach", "implemented programs"
    ],
    "technologies": [
        "IT solutions", "accounting software", "HRIS management", "data-driven decision-making"
    ],
    "reporting": [
        "financial statement analysis", "financial reports", "monthly financial reports", 
        "cost reporting", "compliance"
    ],
    "leadership": [
        "mentored colleagues", "leadership roles", "team development", "managed client needs", 
        "coordinated services", "president", "led projects"
    ],
    "attributes": [
        "meticulous", "attention to detail", "compassionate", "empathy", "collaborative", 
        "commitment to excellence", "dedication", "proactive", "adaptability"
    ],
    "aspirations": [
        "future aspirations", "dedicated to", "focus on process improvements", 
        "commitment to empowering clients"
    ]
}

# Helper function to search for keywords in text
def find_keywords(text, category):
    results = []
    for word in keywords[category]:
        if word in text:
            results.append(word)
    return results

# Function to extract information from text
def extract_information(text):
    print("Extracting information from text...")
    extracted_info = {
        "Name": None,
        "Position Title": None,
        "Years of Experience": None,
        "Industry Experience": None,
        "Key Skills": [],
        "Major Achievements": [],
        "Technological Proficiency": [],
        "Reporting Expertise": None,
        "Communication Skills": None,
        "Leadership Roles": None,
        "Personal Attributes": [],
        "Future Aspirations": None
    }
    
    # Perform NER to identify names and positions
    entities = nlp(text)
    print("Performing NER...")
    for entity in entities:
        if entity['entity'] == 'PER' and not extracted_info['Name']:
            extracted_info['Name'] = entity['word']
        elif entity['entity'] == 'ORG' and not extracted_info['Position Title']:
            extracted_info['Position Title'] = entity['word']
    
    # Custom extraction logic
    if "over a decade" in text or "years of experience" in text:
        extracted_info['Years of Experience'] = "Over a decade"
    
    if "small and medium-sized companies" in text:
        extracted_info['Industry Experience'] = "Small and medium-sized companies"
    
    # Extract key skills
    extracted_info['Key Skills'] = find_keywords(text, "skills")
    
    # Extract major achievements
    extracted_info['Major Achievements'] = find_keywords(text, "achievements")
    
    # Extract technological proficiency
    extracted_info['Technological Proficiency'] = find_keywords(text, "technologies")
    
    # Extract reporting expertise
    if "reporting" in text:
        extracted_info['Reporting Expertise'] = "Yes"
    
    # Extract communication skills
    if "communication" in text:
        extracted_info['Communication Skills'] = "Yes"
    
    # Extract leadership roles
    extracted_info['Leadership Roles'] = find_keywords(text, "leadership")
    
    # Extract personal attributes
    extracted_info['Personal Attributes'] = find_keywords(text, "attributes")
    
    # Extract future aspirations
    extracted_info['Future Aspirations'] = find_keywords(text, "aspirations")
    
    print("Information extracted successfully.")
    return extracted_info

# Function to process all interviewee folders and their recommendation letters
def process_recommendation_letters(main_directory):
    all_extracted_data = []
    
    # Iterate over each folder in the main directory
    for folder_name in os.listdir(main_directory):
        folder_path = os.path.join(main_directory, folder_name)
        
        # Only process directories that follow the interviewee naming convention
        if os.path.isdir(folder_path) and "Recommendation_Letters_of_ID_" in folder_name:
            # Extract interviewee ID from the folder name
            interviewee_id = folder_name.split("_")[-1]
            print(f"Processing interviewee ID: {interviewee_id}")
            
            # Iterate over each text file in the interviewee's folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    
                    # Extract recommender ID from the filename
                    recommender_id = filename.split("_")[-1].replace(".txt", "")
                    print(f"Processing file: {filename} (Recommender ID: {recommender_id})")
                    
                    # Open the file with error handling for encoding
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                            text = file.read()
                            extracted_info = extract_information(text)
                            
                            # Add interviewee and recommender IDs to the extracted information
                            extracted_info['Interviewee ID'] = interviewee_id
                            extracted_info['Recommender ID'] = recommender_id
                            extracted_info['File Name'] = filename
                            
                            all_extracted_data.append(extracted_info)
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")
    print("All files processed successfully.")
    return all_extracted_data

# Function to save extracted data to CSV
def save_to_csv(extracted_data, output_file):
    print(f"Saving data to CSV file: {output_file}")
    # Define the headers for the CSV file
    headers = [
        "File Name", "Interviewee ID", "Recommender ID", "Name", "Position Title", 
        "Years of Experience", "Industry Experience", "Key Skills", "Major Achievements", 
        "Technological Proficiency", "Reporting Expertise", "Communication Skills", 
        "Leadership Roles", "Personal Attributes", "Future Aspirations"
    ]
    
    # Write the data to the CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for data in extracted_data:
            # Convert list fields to a comma-separated string
            data['Key Skills'] = ', '.join(data['Key Skills'])
            data['Major Achievements'] = ', '.join(data['Major Achievements'])
            data['Technological Proficiency'] = ', '.join(data['Technological Proficiency'])
            data['Leadership Roles'] = ', '.join(data['Leadership Roles'])
            data['Personal Attributes'] = ', '.join(data['Personal Attributes'])
            
            writer.writerow(data)
    print("Data saved successfully.")

# Process the recommendation letters
extracted_data = process_recommendation_letters(dir_path)

# Save extracted information to a CSV file in the current "EIGHTFOLD_2" directory
output_csv_file = os.path.join(os.getcwd(), "final_recommendation_letter_to_csv.csv")
save_to_csv(extracted_data, output_csv_file)

print(f"Data extraction and saving to CSV complete. File location: {output_csv_file}")

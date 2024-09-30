import os
import csv
import re
import spacy
from spacy.matcher import PhraseMatcher

# Load the spaCy Transformer model (requires spaCy's 'en_core_web_trf' model)
nlp = spacy.load('en_core_web_trf')

# Combined list of job-related keywords to filter reliable job titles
job_keywords = [
    "manager", "engineer", "analyst", "consultant", "developer", "designer", 
    "specialist", "coordinator", "administrator", "officer", "assistant", 
    "director", "intern", "executive", "technician", "scientist", "representative",
    "architect", "strategist", "supervisor", "planner", "programmer", "researcher", 
    "marketer", "trainer", "advisor", "operator", "controller", "inspector", 
    "facilitator", "producer", "consultant", "auditor", "liaison", "agent", 
    "broker", "buyer", "technologist", "investigator", "instructor",
    # Compound titles
    "managing director", "interim manager", "interim director", "acting manager", "senior manager"
]

# Enhanced list of skills and common phrases
skills_list = [
    "project management", "data analysis", "machine learning", "software development",
    "team leadership", "time management", "programming", "financial analysis", "customer service",
    "problem-solving", "technical writing", "critical thinking", "collaboration", "coding",
    "design", "debugging", "logistics", "operations management", "curriculum design",
    "communication", "innovation", "marketing", "sales", "training"
]

# Create a PhraseMatcher for identifying predefined skills
matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(skill.lower()) for skill in skills_list]
matcher.add("SKILLS", None, *patterns)

# Expanded list of heuristic phrases for job titles
# Expanded list of heuristic phrases for job titles
job_title_phrases = [
    r'\b(applying for|recommended for|seeking position of|interested in|candidate for|fit for a role as|role of|representative)\b (.+?)[,.]',
    r'\b(role as a|position of)\b (.+?)[,.]',
    r'\b(has experience as a|worked as a|served as a)\b (.+?)[,.]',
    r'\b(suitable for|ideal for)\b (.+?)[,.]',
    # New phrases provided by user
    r'\b(applying to the position of|seeking a role as|candidate for the job of|applying for the post of|in pursuit of a career in|aspiring to become a)\b (.+?)[,.]',
    r'\b(a suitable fit for the role of|expressed interest in the position of|a strong applicant for|a valuable addition to your team as a|desires to work as a|a potential hire for the role of)\b (.+?)[,.]',
    r'\b(in consideration for the position of|well-suited for the job of|applying their skills in the field of|seeks to join your company as a|interested in securing the role of|currently applying for positions in)\b (.+?)[,.]',
    r'\b(would excel in the role of|pursuing opportunities in|applying their expertise to the role of|looking to contribute as a|a prospective candidate for|in contention for the job of)\b (.+?)[,.]',
    r'\b(applying their knowledge and skills to|eager to take on the role of|a strong contender for the position of|a natural fit for the role of|recommended for the job of|would make an excellent candidate for)\b (.+?)[,.]',
    # Additional new phrases
    r'\b(a valuable addition to your team as a|desires to work as a|expressed interest in the position of|in pursuit of a career in|candidate for the job of)\b (.+?)[,.]',
    r'\b(in consideration for the position of|would excel in the role of|pursuing opportunities in|applying their expertise to the role of|looking to contribute as a)\b (.+?)[,.]',
    r'\b(applying for the post of|aspiring to become a|currently applying for positions in|a natural fit for the role of)\b (.+?)[,.]'
]


# Generic phrases for positive feedback
generic_phrases = [
    "great potential", "incredible enthusiasm", "positive attitude", "hard-working", 
    "good communication skills", "quick learner", "outstanding individual", 
    "remarkable dedication", "excellent team player", "highly motivated", "strong work ethic",
    "creative thinker", "exceptional performance", "adaptable and flexible", "natural leader",
    "committed professional", "proven track record", "demonstrated excellence", 
    "passionate about", "enthusiastic learner", "results-driven", "skilled professional",
    "asset to any team", "goes above and beyond", "driven to succeed", "dedicated worker",
    "valuable member", "a pleasure to work with", "takes initiative", "reliable and trustworthy",
    "exceeds expectations", "strong sense of responsibility", "proactive and diligent",
    "works well under pressure", "highly dependable", "motivated self-starter",
    "highly regarded by peers", "impressive skill set", "natural talent", "team-oriented mindset",
    "excellent rapport with clients", "versatile in skill set", "continual improvement",
    "committed to growth", "admirable qualities", "exceptional interpersonal skills",
    "always willing to help", "willing to take on challenges", "a true professional",
    "consistently delivers quality work"
]

# Custom list of common nouns to exclude as non-skills
exclude_words = {
    "individual", "project", "environment", "needs", "years", "focus", "roles", "satisfaction",
    "relationships", "education", "dedication", "position", "background", "outcomes", "impact",
    "training", "design", "abilities", "field", "management", "team", "members", "service",
    "clients", "students", "experience", "setting", "work", "success"
}

# Helper function to clean extracted job titles
def clean_job_title(job_title_candidate, recommender_job=None):
    # Remove unwanted words and phrases that are not job titles
    unwanted_phrases = [
        "driving", "managing tasks", "working on", "engaging in", "overseeing", "ensuring", 
        "facilitating", "achieving", "liaison", "leader", "effective", "efficient"
    ]
    
    # List of phrases indicating recommender's own role
    recommender_exclusions = [
        "yoga instructor", "teacher", "director", "manager", "coach", "engineer",
        "instructor", "consultant", "former", "previously", "my role as", "i am", "i work as", "in my capacity as"
    ]

    if not job_title_candidate:
        return None

    # Check if the job title contains unwanted phrases
    if any(phrase in job_title_candidate.lower() for phrase in unwanted_phrases):
        return None

    # Remove phrases like "a," "an," "the," and unnecessary words
    if job_title_candidate.lower() in ["a", "an", "the", "their"]:
        return None

    # Check if the job title is similar to the recommender's job
    if recommender_job and recommender_job.lower() in job_title_candidate.lower():
        return None

    # Exclude phrases that seem like recommender-specific roles
    if any(phrase in job_title_candidate.lower() for phrase in recommender_exclusions):
        return None

    # Allow titles containing keywords or compound titles
    if any(keyword in job_title_candidate.lower() for keyword in job_keywords):
        return job_title_candidate.strip()

    # If none of the keywords match, consider it an invalid title
    return None

def extract_job_title(doc, recommender_job=None):
    # Define indicators for the recommender's role
    recommender_indicators = [
        "I am", "As a", "In my role", "I work as", "I serve as", "Having worked as",
        "Previously served as", "former", "my position as", "During my time as", "recommendation comes from"
    ]
    
    # Apply job title phrases specifically related to the interviewee's job application
    for pattern in job_title_phrases:
        matches = re.findall(pattern, doc.text, re.IGNORECASE)
        if matches:
            job_title_candidate = clean_job_title(matches[0][1].strip(), recommender_job)
            # Ensure the job title contains at least one keyword from the job_keywords list
            if job_title_candidate and any(keyword in job_title_candidate.lower() for keyword in job_keywords):
                return job_title_candidate  # Return immediately if a prioritized phrase is found

    # Fallback to using noun chunks as possible job titles if heuristic phrases fail
    for chunk in doc.noun_chunks:
        # Skip noun chunks that contain recommender-specific phrases
        if any(indicator.lower() in chunk.text.lower() for indicator in recommender_indicators):
            continue
        
        if chunk.root.dep_ in {"attr", "dobj", "pobj"} and chunk.root.pos_ == "NOUN":
            # Further filter to exclude common words that aren't job titles
            job_title_candidate = clean_job_title(chunk.text.strip(), recommender_job)
            if job_title_candidate and all(word.lower() not in exclude_words for word in job_title_candidate.split()):
                # Ensure the chunk contains a job-related keyword
                if any(keyword in job_title_candidate.lower() for keyword in job_keywords):
                    return job_title_candidate

    # If no valid job title is found, return None
    return None

# Expanded list of patterns for past job experience
past_job_phrases = [
    r'\b(worked as a|served as a|held the position of|experience in|worked in the capacity of|formerly an|held a role as)\b (.+?)[,.]',
    r'\b(previously worked as a|previous experience as a|had a position as|past role as|engaged as a)\b (.+?)[,.]',
    r'\b(former role as|prior experience in the role of|past experience as a)\b (.+?)[,.]',
    r'\b(held the title of|undertook the role of|was responsible for|previously occupied the role of|functioned as|served in the capacity of)\b (.+?)[,.]',
    r'\b(gained experience as a|developed skills as a|known for their work as a|prior engagement as a|has a background in|proven experience in the role of)\b (.+?)[,.]',
    r'\b(has served as|had a successful stint as|spent time as|fulfilled the role of|known to have excelled as a)\b (.+?)[,.]',
    r'\b(led projects as a|was a part of the team as a|demonstrated expertise as a|was employed as|functioned in the capacity of)\b (.+?)[,.]',
    # New patterns to capture compound titles
    r'\b(interim|acting|senior|managing)\b (.+?)[,.]',
]


def extract_past_experience(doc):
    # Extract past job experiences using defined patterns
    for pattern in past_job_phrases:
        matches = re.findall(pattern, doc.text, re.IGNORECASE)
        if matches:
            experience_candidate = matches[0][1].strip()
            # Clean and validate the extracted experience
            cleaned_experience = clean_job_title(experience_candidate)
            if cleaned_experience:
                return cleaned_experience
    return None

def extract_information_from_text(text):
    # Debug statement to indicate start of text processing
    print("Extracting information from text...")

    # Run spaCy NLP processing with transformer model
    doc = nlp(text)

    # Extract job of the recommender
    recommender_job = None
    recommender_job_matches = re.findall(r'\b(I am a|I am an|As a|In my role as a|I work as a|I serve as a) (.+?)[,.]', text, re.IGNORECASE)
    if recommender_job_matches:
        recommender_job = recommender_job_matches[0][1]
        print(f"Recommender Job Found: {recommender_job}")

    # Extract job title using the enhanced method
    job_title = extract_job_title(doc, recommender_job)
    print(f"Job Title Found: {job_title if job_title else 'None'}")

    # Extract past job experience using stricter filtering
    past_experience = extract_past_experience(doc)
    print(f"Past Experience Found: {past_experience if past_experience else 'None'}")

    # Extract skills using PhraseMatcher and dependency parsing
    skills_vouched_for = set()
    matches = matcher(doc)
    for match_id, start, end in matches:
        skills_vouched_for.add(doc[start:end].text)

    # Further identify skills using adjectives and nouns pairs
    for token in doc:
        if token.pos_ in {'NOUN', 'ADJ'} and token.text.lower() not in exclude_words and len(token.text) > 2:
            if token.dep_ in {'nsubj', 'dobj', 'attr', 'pobj'}:
                skill_phrase = " ".join([child.text for child in token.children if child.dep_ == 'amod']) + " " + token.text
                skills_vouched_for.add(skill_phrase.strip())

    print(f"Skills Vouched For: {', '.join(skills_vouched_for)}")

    # Extract generic positive phrases
    phrases_found = [phrase for phrase in generic_phrases if re.search(r'\b' + re.escape(phrase.lower()) + r'\b', text.lower())]

    # Return extracted data
    return {
        "Job Title applying to": job_title or '',
        "Past Job Experience": past_experience or '',
        "Skills Vouched for": ', '.join(skills_vouched_for),
        "Phrases": phrases_found,
        "Recommender Job": recommender_job or ''
    }

def process_recommendation_letters(root_directory='Final_Recommendation_Letters'):
    csv_data = []
    
    # Debug statement to indicate start of directory traversal
    print(f"Processing recommendation letters in directory: {root_directory}")
    
    # Traverse the directory structure
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith('Recommendation_Letters_of_ID_'):
            interviewee_id = folder_name.split('_')[-1]
            print(f"\nProcessing folder: {folder_name} (Interviewee ID: {interviewee_id})")
            
            # Process each recommendation letter in the folder
            for file_name in os.listdir(folder_path):
                if file_name.startswith('Recommendation_From_ID_'):
                    recommender_id = file_name.split('_')[-1].replace('.txt', '')
                    file_path = os.path.join(folder_path, file_name)
                    print(f"  Processing file: {file_name} (Recommender ID: {recommender_id})")
                    
                    # Read the content of the recommendation letter with encoding handling
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                            text = file.read()
                    except Exception as e:
                        print(f"  Error reading file {file_path}: {e}")
                        continue
                    
                    # Extract information using advanced NLP
                    extracted_info = extract_information_from_text(text)
                    
                    # Prepare the row for the CSV
                    csv_data.append({
                        'Interviewee ID': interviewee_id,
                        'Recommender ID': recommender_id,
                        'Job Title applying to': extracted_info.get('Job Title applying to', ''),
                        'Past Job Experience': extracted_info.get('Past Job Experience', ''),
                        'Skills Vouched for': extracted_info.get('Skills Vouched for', ''),
                        'Phrases': ', '.join(extracted_info.get('Phrases', [])),
                        'Recommender Job': extracted_info.get('Recommender Job', '')
                    })
    
    # Write data to CSV file
    output_csv = 'Recommendation_Letter_CSV.csv'
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['Interviewee ID', 'Recommender ID', 'Job Title applying to', 'Past Job Experience', 'Skills Vouched for', 'Phrases', 'Recommender Job']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        print(f"\nOutput CSV file created: {output_csv}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

# Run the processing function within the default directory structure
process_recommendation_letters()


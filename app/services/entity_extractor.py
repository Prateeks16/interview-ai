from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Set, Optional
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Hugging Face NER Model ---

NER_MODEL_NAME = "elastic/distilbert-base-uncased-finetuned-conll03-english"


ner_pipeline = None

def get_ner_pipeline():
    global ner_pipeline
    if ner_pipeline is None:
        try:
            logger.info(f"Loading NER model: {NER_MODEL_NAME}")
            # Explicitly load tokenizer and model to handle potential trust issues if any
            tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
            model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
            ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
            logger.info("NER model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            # Fallback or raise error
            ner_pipeline = "ERROR" # Mark as failed
    if ner_pipeline == "ERROR":
        return None # Don't attempt to use if loading failed
    return ner_pipeline

# --- Basic Skill Keyword List (Fallback or Augmentation) ---
# This is a very simplified list. A more comprehensive list or ontology would be better.
COMMON_SKILLS_KEYWORDS = {
    "python", "java", "c++", "javascript", "react", "angular", "vue", "node.js", "django", "flask",
    "spring", "ruby", "rails", "php", "laravel", "swift", "kotlin", "android", "ios", "html", "css",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "docker", "kubernetes", "aws", "azure", "gcp",
    "linux", "git", "rest", "graphql", "api", "microservices", "machine learning", "deep learning",
    "data analysis", "data science", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "communication", "teamwork", "problem-solving", "agile", "scrum", "project management",
    "leadership", "fintech", "blockchain", "cybersecurity"
}

def extract_entities_from_text(text: str, sections: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
 
    extracted_data = {
        "skills": [],
        "experience_keywords": [], 
        "organizations": [],
        "locations": [],
        "persons": [],
        "misc_entities": [] 
    }

    # 1. NER Extraction
    pipeline_instance = get_ner_pipeline()
    if pipeline_instance:
        try:
            # Process the whole text or specific sections if provided
            text_to_process = sections.get("skills", "") + "\n" + sections.get("experience", text) if sections else text
            
        
            max_chunk_length = 400 
            chunks = []
            if len(text_to_process.split()) > max_chunk_length:
                current_chunk = ""
                for sentence in re.split(r'(?<=[.!?])\s+', text_to_process): # Split by sentences
                    if len(current_chunk.split()) + len(sentence.split()) <= max_chunk_length:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk: chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                if current_chunk: chunks.append(current_chunk.strip())
            else:
                chunks = [text_to_process]

            ner_results = []
            for chunk in chunks:
                if chunk: # Ensure chunk is not empty
                    ner_results.extend(pipeline_instance(chunk))
            
            # Process NER results
            current_skills = set()
            for entity in ner_results:
                entity_group = entity.get("entity_group")
                word = entity.get("word", "").strip()
                if not word: continue

                
               
                if entity_group in ["ORG", "NORP"]: # NORP is nationalities or religious or political groups
                    extracted_data["organizations"].append(word)
                elif entity_group in ["LOC", "GPE"]: # GPE is Geo-Political Entity
                    extracted_data["locations"].append(word)
                elif entity_group == "PER":
                    extracted_data["persons"].append(word)
                elif entity_group == "MISC": # Miscellaneous can sometimes include skills/technologies
                    # Check if MISC entity looks like a common skill
                    if word.lower() in COMMON_SKILLS_KEYWORDS:
                        current_skills.add(word.lower())
                    else:
                        extracted_data["misc_entities"].append(word)
                # If the model has specific skill tags (unlikely for general NER), use them
                elif "SKILL" in entity_group.upper() or "TECH" in entity_group.upper():
                     current_skills.add(word.lower())
            
            extracted_data["skills"].extend(list(current_skills))

        except Exception as e:
            logger.error(f"Error during NER processing: {e}")
            # Fallback to keyword matching if NER fails or gives no skills

    # 2. Keyword-based skill extraction (augment or fallback)
    
    
    # Process text from 'skills' section first if available, then whole text

    text_for_skill_keywords = sections.get("skills", "").lower() if sections else ""
    if not text_for_skill_keywords or len(extracted_data["skills"]) < 5 : # If skill section is empty or few skills found
        text_for_skill_keywords += "\n" + text.lower() # Add whole resume text

    found_keyword_skills = set(extracted_data["skills"]) # Initialize with NER skills
    # Use regex to find whole words to avoid matching substrings within words
    for skill in COMMON_SKILLS_KEYWORDS:
        # Regex for whole word match, case insensitive. Handle special chars like C++
        # Simple word boundary for now.
        # For "c++", regex might need to be specific: re.escape(skill)
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_for_skill_keywords, re.IGNORECASE):
            found_keyword_skills.add(skill)
    
    extracted_data["skills"] = sorted(list(found_keyword_skills))


    # Rudimentary experience keyword extraction
    if sections and "experience" in sections:
        experience_text = sections["experience"].lower()
        exp_keywords = set()
        for skill in COMMON_SKILLS_KEYWORDS: # Check common skills in experience
             pattern = r'\b' + re.escape(skill) + r'\b'
             if re.search(pattern, experience_text, re.IGNORECASE):
                exp_keywords.add(skill)
        # Also add some MISC entities found by NER if they appear in experience text
        for misc_entity in extracted_data["misc_entities"]:
            if misc_entity.lower() in experience_text:
                exp_keywords.add(misc_entity) # Assuming misc entities can be tech/tools
        extracted_data["experience_keywords"] = sorted(list(exp_keywords))

    # Deduplicate lists
    for key in extracted_data:
        if isinstance(extracted_data[key], list):
            # Maintain order for skills if possible, for others set conversion is fine
            if key == "skills" or key == "experience_keywords":
                 # Simple deduplication while trying to preserve order
                seen = set()
                extracted_data[key] = [x for x in extracted_data[key] if not (x in seen or seen.add(x))]
            else:
                extracted_data[key] = sorted(list(set(extracted_data[key])))
                
    return extracted_data

def extract_skills_from_jd(jd_text: str) -> List[str]:
    """
    A simplified function to extract potential skills from a job description.
    Can be enhanced with NER or more sophisticated NLP.
    """
    jd_text_lower = jd_text.lower()
    found_skills = set()
    
    # Try NER on JD
    pipeline_instance = get_ner_pipeline()
    if pipeline_instance:
        try:
            # Simplified chunking for JD as well
            max_chunk_length = 400 
            chunks = []
            if len(jd_text_lower.split()) > max_chunk_length:
                current_chunk = ""
                for sentence in re.split(r'(?<=[.!?])\s+', jd_text_lower): # Split by sentences
                    if len(current_chunk.split()) + len(sentence.split()) <= max_chunk_length:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk: chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                if current_chunk: chunks.append(current_chunk.strip())

            else:
                chunks = [jd_text_lower]

            ner_results_jd = []
            for chunk in chunks:
                if chunk:
                    ner_results_jd.extend(pipeline_instance(chunk))

            for entity in ner_results_jd:
                word = entity.get("word", "").strip().lower()
                # If MISC or specific skill tags (if model supports)
                if entity.get("entity_group") == "MISC" and word in COMMON_SKILLS_KEYWORDS:
                    found_skills.add(word)
                elif "SKILL" in entity.get("entity_group", "").upper() or "TECH" in entity.get("entity_group", "").upper():
                    found_skills.add(word)
        except Exception as e:
            logger.warning(f"NER on JD failed or gave partial results: {e}")
            # Fallback to keyword if NER has issues

    # Keyword matching (primary or fallback)
    for skill in COMMON_SKILLS_KEYWORDS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, jd_text_lower, re.IGNORECASE):
            found_skills.add(skill)
            
    return sorted(list(found_skills))

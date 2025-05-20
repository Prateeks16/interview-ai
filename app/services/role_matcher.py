from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Sentence Transformer Model ---
# Using a lightweight, effective model.
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
sentence_model = None

def get_sentence_transformer_model():
    global sentence_model
    if sentence_model is None:
        try:
            logger.info(f"Loading sentence transformer model: {SENTENCE_MODEL_NAME}")
            sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
            logger.info("Sentence transformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            sentence_model = "ERROR"
    if sentence_model == "ERROR":
        return None
    return sentence_model

def calculate_tfidf_similarity(resume_text: str, jd_text: str) -> Optional[float]:
    """Calculates TF-IDF cosine similarity between resume text and job description."""
    if not resume_text or not jd_text:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        # cosine_similarity returns a matrix, get the specific similarity value
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity_score)
    except Exception as e:
        logger.error(f"Error calculating TF-IDF similarity: {e}")
        return None


def calculate_semantic_similarity(resume_text: str, jd_text: str) -> Optional[float]:
    """Calculates semantic similarity using Sentence Transformers."""
    if not resume_text or not jd_text:
        return 0.0
        
    model = get_sentence_transformer_model()
    if not model:
        logger.warning("Sentence transformer model not available for semantic similarity.")
        return None
    try:
        # It's better to embed meaningful chunks or whole documents if not too long.
        # For simplicity, embedding the whole texts. Consider chunking for very long texts.
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        jd_embedding = model.encode(jd_text, convert_to_tensor=True)
        
        # Compute cosine-similarity
        semantic_sim_score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
        return float(semantic_sim_score)
    except Exception as e:
        logger.error(f"Error calculating semantic similarity: {e}")
        return None

def compare_skills(
    resume_skills: List[str], 
    jd_skills_extracted: List[str] # Skills already extracted from JD
    ) -> Tuple[List[str], List[str]]:
    """
    Compares skills from resume against skills mentioned or inferred from JD.
    Returns (matched_skills, missing_skills_from_jd).
    """
    if not resume_skills and not jd_skills_extracted:
        return [], []
    if not jd_skills_extracted: # If JD has no skills, all resume skills are "not in JD" but none are "missing"
        return [], [] 
    if not resume_skills: # If resume has no skills, all JD skills are "missing"
        return [], sorted(list(set(s.lower() for s in jd_skills_extracted)))

    resume_skills_set = set(s.lower() for s in resume_skills)
    jd_skills_set = set(s.lower() for s in jd_skills_extracted)

    matched = sorted(list(resume_skills_set.intersection(jd_skills_set)))
    missing_from_jd = sorted(list(jd_skills_set.difference(resume_skills_set)))
    # resume_only = sorted(list(resume_skills_set.difference(jd_skills_set)))
    
    return matched, missing_from_jd #, resume_only
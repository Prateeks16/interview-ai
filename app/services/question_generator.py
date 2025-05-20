from typing import List, Dict, Optional
import random
import logging
import re
import os

# --- Hugging Face T5 ---
from transformers import T5ForConditionalGeneration, T5Tokenizer

# --- Import entity_extractor for jd_keywords ---
from ..services import entity_extractor

# --- Google Gemini (Commented Out - for future integration) ---
# import google.generativeai as genai

logger = logging.getLogger(__name__)

# --- Configuration ---
USE_GEMINI_API_IF_AVAILABLE = os.getenv("USE_GEMINI_API", "false").lower() == "true"
GOOGLE_API_KEY = os.getenv("")

# --- T5 Model Components ---
T5_MODEL_NAME = os.getenv("T5_MODEL_FOR_QG", "t5-base")
t5_tokenizer = None # Global for T5 tokenizer
t5_model = None     # Global for T5 model

# --- Gemini Model Components (Commented Out) ---
# gemini_model_instance = None
# GEMINI_MODEL_NAME = "gemini-1.0-pro-latest"

def _initialize_t5():
    """Initializes the T5 model and tokenizer if not already done."""
    global t5_tokenizer, t5_model
    if t5_tokenizer is None or t5_model is None:
        if t5_tokenizer == "ERROR": # Don't retry if loading previously failed
            logger.warning("T5 model loading previously failed. Not retrying.")
            return False
        try:
            logger.info(f"Loading T5 QG model and tokenizer: {T5_MODEL_NAME}")
            t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
            t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)
            logger.info("T5 QG model and tokenizer loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load T5 QG model ({T5_MODEL_NAME}): {e}", exc_info=True)
            t5_tokenizer = "ERROR" 
            t5_model = "ERROR"
            return False
    elif t5_tokenizer == "ERROR" or t5_model == "ERROR": # Already marked as error
        return False
    return True # Already loaded successfully

# --- (Commented Out) Gemini Initialization ---
# def _initialize_gemini():
#     global gemini_model_instance
#     if not GOOGLE_API_KEY:
#         logger.warning("GOOGLE_API_KEY not found in environment. Gemini API will not be available.")
#         return False
#     if gemini_model_instance is None: 
#         if gemini_model_instance == "ERROR":
#             return False
#         try:
#             logger.info(f"Initializing Gemini Pro model: {GEMINI_MODEL_NAME}")
#             import google.generativeai as genai # Import here to avoid error if not installed
#             genai.configure(api_key=GOOGLE_API_KEY)
#             gemini_model_instance = genai.GenerativeModel(GEMINI_MODEL_NAME)
#             logger.info("Gemini Pro model initialized successfully.")
#             return True
#         except Exception as e:
#             logger.error(f"Failed to initialize Gemini Pro model: {e}", exc_info=True)
#             gemini_model_instance = "ERROR"
#             return False
#     elif gemini_model_instance == "ERROR":
#         return False
#     return True

def _is_meaningful_term(term: str, min_len: int = 3) -> bool:
    """Helper to filter out very short or common noisy terms."""
    if not term or len(term) < min_len:
        return False
    common_fillers = {"and", "the", "for", "with", "are", "dev", "etc", "various", "based", "also", "this", "that", "job", "role"}
    if term.lower() in common_fillers:
        return False
    # Check if term is purely numeric
    if term.isdigit():
        return False
    return True

def _parse_llm_generated_text_into_questions(generated_text: str, num_questions_target: int) -> List[str]:
    """Parses raw text from an LLM into a list of distinct questions."""
    questions = []
    if not generated_text or not generated_text.strip():
        logger.warning("LLM generated empty or whitespace-only text.")
        return []
        
    potential_lines = generated_text.split('\n')
    for line in potential_lines:
        line_cleaned = line.strip()
        line_cleaned = re.sub(r"^\s*(\d+\.|[a-zA-Z][.)]|\*|-)\s*", "", line_cleaned).strip() # Remove list markers
        if line_cleaned and len(line_cleaned) > 10 and line_cleaned.endswith('?'):
            questions.append(line_cleaned)

    if not questions and len(generated_text) > 20: # If newline splitting failed, try sentence splitting
        logger.info("Newline splitting yielded no questions, trying regex sentence splitting.")
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', generated_text)
        for sentence in sentences:
            s_cleaned = sentence.strip()
            s_cleaned = re.sub(r"^\s*(\d+\.|[a-zA-Z][.)]|\*|-)\s*", "", s_cleaned).strip()
            if s_cleaned and len(s_cleaned) > 10 and s_cleaned.endswith('?'):
                if s_cleaned not in questions: questions.append(s_cleaned)
    
    seen = set()
    unique_questions = [q for q in questions if not (q in seen or seen.add(q))] # Ensure uniqueness
    
    logger.info(f"Parsed {len(unique_questions)} unique questions from LLM output.")
    return unique_questions[:num_questions_target]

def _generate_questions_with_t5_internal(
    prompt_text: str,
    num_questions_target: int,
    max_length_generation: int = 250 
) -> List[str]:
    if not _initialize_t5() or t5_tokenizer == "ERROR" or t5_model == "ERROR":
        logger.warning("T5 model not available or failed to load. Cannot use T5 for QG.")
        return []

    try:
        inputs = t5_tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True, padding="longest")
        summary_ids = t5_model.generate(
            inputs.input_ids,
            num_beams=5,
            max_length=max_length_generation,
            early_stopping=True,
            no_repeat_ngram_size=3, # Prevents overly repetitive phrases
            # temperature=0.7, # Can uncomment for more varied (but potentially less coherent) output
            # top_p=0.9,       # For nucleus sampling
        )
        generated_text = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        logger.info(f"T5 Raw Generated Text:\n'''{generated_text}'''") # Log raw output clearly
        return _parse_llm_generated_text_into_questions(generated_text, num_questions_target)
    except Exception as e:
        logger.error(f"Error during T5 question generation: {e}", exc_info=True)
        return []

# --- (Commented Out) Gemini Question Generation Logic ---
# def _generate_questions_with_gemini_internal(prompt_text: str, num_questions_target: int) -> List[str]:
#     if not _initialize_gemini() or gemini_model_instance == "ERROR" or gemini_model_instance is None:
#         logger.warning("Gemini API not configured/initialized. Cannot use Gemini for QG.")
#         return []
#     try:
#         # import google.generativeai as genai # Ensure import is within try if not global
#         response = gemini_model_instance.generate_content(prompt_text)
#         if not response.candidates or not response.candidates[0].content.parts: # Check for blocked/empty
#             # Log prompt feedback for safety reasons
#             safety_ratings_str = ""
#             if response.prompt_feedback and response.prompt_feedback.safety_ratings:
#                 safety_ratings_str = ", ".join([f"{rating.category}: {rating.probability.name}" for rating in response.prompt_feedback.safety_ratings])
#             logger.warning(f"Gemini response was empty or blocked. Prompt Feedback: Blocked - {response.prompt_feedback.block_reason if response.prompt_feedback else 'N/A'}, Safety Ratings - [{safety_ratings_str}]")
#             return []
#         generated_text = response.text
#         logger.info(f"Gemini Raw Generated Text:\n'''{generated_text}'''")
#         return _parse_llm_generated_text_into_questions(generated_text, num_questions_target)
#     except Exception as e:
#         logger.error(f"Error calling Gemini API or processing its response: {e}", exc_info=True)
#         return []

# In app/services/question_generator.py

def _craft_llm_prompt(
    resume_summary: str,
    jd_keywords: List[str],
    candidate_skills: List[str],
    role_title: Optional[str],
    num_questions_target: int,
    model_type: str = "t5"
) -> str:
    """Crafts a detailed prompt for the LLM."""
    filtered_candidate_skills = [s for s in candidate_skills if _is_meaningful_term(s)][:3] # Max 3 skills
    filtered_jd_keywords = [k for k in jd_keywords if _is_meaningful_term(k)][:3]     # Max 3 JD keywords

    # --- Option 1: Stronger initial instruction, less resume focus ---
    prompt_lines = [
        f"Your task is to generate exactly {num_questions_target} distinct interview questions.",
        "These questions should assess a candidate's suitability for a job."
    ]
    if role_title:
        prompt_lines.append(f"The job role is: '{role_title}'.")
    
    context_parts = []
    if filtered_candidate_skills:
        context_parts.append(f"candidate skills include {', '.join(filtered_candidate_skills)}")
    if filtered_jd_keywords:
        context_parts.append(f"job requires {', '.join(filtered_jd_keywords)}")
    
    # Drastically shorten or even temporarily remove resume snippet to see if it helps
    if resume_summary and context_parts: # Only add resume if other context exists
        resume_snippet = " ".join(resume_summary.split()[:30]) # VERY short snippet, e.g., 30 words
        context_parts.append(f"resume mentions \"{resume_snippet}\"")
    
    if context_parts:
        prompt_lines.append(f"Consider the following context: {'; '.join(context_parts)}.")
    else:
        prompt_lines.append("Generate general technical and behavioral interview questions.")


    prompt_lines.append("\nStrictly follow these output instructions:")
    prompt_lines.append(f"- Output exactly {num_questions_target} questions.")
    prompt_lines.append("- Each question must be on its own new line.")
    prompt_lines.append("- Every question must end with a single question mark (?).")
    prompt_lines.append("- Do NOT output any other text, greetings, or explanations. Only the questions.")
    
    final_prompt = "\n".join(prompt_lines)
    logger.debug(f"Crafted LLM Prompt ({model_type}):\n'''{final_prompt}'''")
    return final_prompt

def generate_llm_interview_questions(
    resume_text: str,
    job_description_text: str,
    extracted_skills: List[str], 
    role_title: Optional[str] = None,
    num_questions: int = 5 
) -> List[str]:
    logger.info(f"Generating {num_questions} LLM interview questions for role: {role_title or 'N/A'}")
    
    resume_summary = " ".join(resume_text.split()[:150]) # Summary for prompt crafting
    
    # Use entity_extractor to get skills/keywords from JD
    jd_extracted_keywords_all = entity_extractor.extract_skills_from_jd(job_description_text)
    logger.debug(f"JD keywords for prompt context (raw from entity_extractor): {jd_extracted_keywords_all[:10]}")

    # Filter skills from resume for prompt crafting
    filtered_candidate_skills_for_prompt = [s for s in extracted_skills if _is_meaningful_term(s)]

    questions = []
    
    # --- Primary: T5 Generation ---
    logger.info("Attempting question generation with T5 model.")
    t5_prompt = _craft_llm_prompt(
        resume_summary, 
        jd_extracted_keywords_all, # Pass all extracted JD keywords to prompt crafter
        filtered_candidate_skills_for_prompt, 
        role_title, 
        num_questions, 
        model_type="t5"
    )
    questions = _generate_questions_with_t5_internal(t5_prompt, num_questions)
    
    if questions:
        logger.info(f"Successfully generated {len(questions)} questions with T5.")
        return questions[:num_questions] # Ensure correct number of questions

    # --- (Commented Out) Fallback: Gemini Generation ---
    # This block is reached if T5 fails (questions list is empty)
    # logger.info("T5 generation failed or produced no questions. Checking Gemini fallback availability.")
    # use_gemini = USE_GEMINI_API_IF_AVAILABLE and GOOGLE_API_KEY
    # if use_gemini:
    #     if not _initialize_gemini():
    #         logger.warning("Gemini could not be initialized. Cannot use as fallback.")
    #     else:
    #         logger.info("Attempting question generation with Gemini model as fallback.")
    #         gemini_prompt = _craft_llm_prompt(
    #             resume_summary, 
    #             jd_extracted_keywords_all, 
    #             filtered_candidate_skills_for_prompt, 
    #             role_title, 
    #             num_questions, 
    #             model_type="gemini"
    #         )
    #         questions = _generate_questions_with_gemini_internal(gemini_prompt, num_questions)
    #         if questions and not (len(questions) == 1 and "Error with Gemini" in questions[0]): # Check for actual questions
    #             logger.info(f"Successfully generated {len(questions)} questions with Gemini as fallback.")
    #             return questions[:num_questions]
    #         else:
    #             logger.warning("Gemini fallback also failed or produced no valid questions.")
    
    # If all LLM attempts fail
    logger.warning("All LLM attempts (T5 and/or Gemini) failed to generate questions. Returning an error message.")
    return ["Failed to generate questions using LLMs. Please check model configurations and input data."]

def preload_llm_models():
    """
    Preloads LLM models (currently T5) for question generation.
    This function is intended to be called by main.py at startup.
    """
    logger.info("Preloading LLM models for question generation...")
    initialized_t5 = _initialize_t5()
    if initialized_t5:
        logger.info("T5 model preloading attempt successful (or already loaded).")
    else:
        logger.warning("T5 model preloading attempt failed.")
    
    # --- (Commented out) Gemini Preloading ---
    # use_gemini_for_preload = USE_GEMINI_API_IF_AVAILABLE and GOOGLE_API_KEY
    # if use_gemini_for_preload:
    #     logger.info("Attempting to preload/initialize Gemini model...")
    #     initialized_gemini = _initialize_gemini()
    #     if initialized_gemini:
    #         logger.info("Gemini model preloading/initialization attempt successful (or already done).")
    #     else:
    #         logger.warning("Gemini model preloading/initialization attempt failed.")
            
    logger.info("LLM model preloading process complete.")
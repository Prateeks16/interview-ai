from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Body
from typing import List, Optional
import logging
import random 

from ..models import schemas 
from ..services import (
    resume_parser,
    entity_extractor,
    role_matcher,
    question_generator,
    transcription_service, 
    video_analysis_service, 
    feedback_service 
)
from ..utils import file_handling

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Resume Processing Endpoints ---
@router.post("/resumes/upload/", response_model=schemas.ResumeParseResponse)
async def upload_and_parse_resume(resume_file: UploadFile = File(...)):
    """
    Uploads a resume file (.pdf or .docx), parses it to text,
    and extracts named entities (skills, etc.).
    """
    if not (resume_file.filename.lower().endswith(".pdf") or resume_file.filename.lower().endswith(".docx")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or DOCX.")

    try:
        # 1. Read the entire file content into memory first
        
        file_content_bytes = await resume_file.read()
        # After reading, the UploadFile's internal pointer is at the end.

        # 2. Rewind the UploadFile stream to the beginning so it can be read again for saving.
       
        await resume_file.seek(0)

        # 3. Save the file to disk using the (now rewound) UploadFile stream.
        
        file_path_on_disk = await file_handling.save_upload_file(resume_file, file_handling.UPLOADS_DIR_RESUMES)
        logger.info(f"Resume uploaded and saved to: {file_path_on_disk}")
        

        # 4. Parse Resume Text using the 'file_content_bytes' we read into memory earlier.
        extracted_text, error = resume_parser.extract_text_from_resume(resume_file.filename, file_content_bytes)
        if error:
            logger.error(f"Error parsing resume {resume_file.filename}: {error}")
            raise HTTPException(status_code=422, detail=f"Error parsing resume: {error}")
        if not extracted_text:
            logger.warning(f"No text extracted from {resume_file.filename}")
            raise HTTPException(status_code=422, detail="Could not extract text from resume.")

        # 5. Basic Sectioning (Optional, can help NER)
        resume_sections = resume_parser.extract_sections_from_text(extracted_text)
        logger.info(f"Extracted sections: {list(resume_sections.keys())}")


        # 6. Extract Entities (Skills, etc.)
       
        extracted_entities_dict = entity_extractor.extract_entities_from_text(extracted_text, sections=resume_sections)
        
        entities_schema = schemas.ExtractedEntities(
            skills=extracted_entities_dict.get("skills", []),
            experience_sections=extracted_entities_dict.get("experience_keywords", [])
        )
        logger.info(f"Extracted entities for {resume_file.filename}: Skills - {len(entities_schema.skills)}")

        return schemas.ResumeParseResponse(
            filename=resume_file.filename,
            text_content=extracted_text,
            entities=entities_schema,
            message="Resume uploaded and parsed successfully."
        )

    except HTTPException as http_exc:
        raise http_exc # Re-raise FastAPI's HTTP exceptions
    except ValueError as ve: # Catch specific errors from parsers or seek/read if file is already closed unexpectedly
        logger.error(f"ValueError during resume processing: {ve}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during resume upload/parse for {resume_file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@router.post("/resumes/compare/", response_model=schemas.CompareWithJDResponse)
async def compare_resume_with_job_description(request: schemas.CompareWithJDRequest):
    """
    Compares resume content (text and extracted skills) with a job description.
    Provides TF-IDF similarity, semantic similarity, and skill matching.
    """
    try:
        # 1. Extract skills from Job Description
        jd_skills = entity_extractor.extract_skills_from_jd(request.job_description_text)
        logger.info(f"Skills extracted from JD: {jd_skills}")

        # 2. Calculate TF-IDF Similarity (Resume Text vs JD Text)
        tfidf_score = role_matcher.calculate_tfidf_similarity(
            request.resume_text,
            request.job_description_text
        )

        # 3. Calculate Semantic Similarity (Resume Text vs JD Text)
        semantic_score = role_matcher.calculate_semantic_similarity(
            request.resume_text,
            request.job_description_text
        )

        # 4. Compare Resume Skills with JD Skills
        matched_skills, missing_skills = role_matcher.compare_skills(
            resume_skills=request.extracted_skills,
            jd_skills_extracted=jd_skills
        )

        comparison_result = schemas.ComparisonResult(
            tfidf_similarity_score=tfidf_score,
            semantic_similarity_score=semantic_score,
            matched_skills=matched_skills,
            missing_skills_from_jd=missing_skills
        )

        return schemas.CompareWithJDResponse(
            comparison=comparison_result,
            message="Resume and JD comparison complete."
        )
    except Exception as e:
        logger.error(f"Error during resume/JD comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during comparison: {e}")

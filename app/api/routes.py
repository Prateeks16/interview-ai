from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Body
from typing import List, Optional
import logging
import random # For dummy video_id in stub

from ..models import schemas # Use .. to go up one level to app, then models
from ..services import (
    resume_parser,
    entity_extractor,
    role_matcher,
    question_generator,
    transcription_service, # Stub
    video_analysis_service, # Stub
    feedback_service # Stub
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
        # This is for parsing.
        file_content_bytes = await resume_file.read()
        # After reading, the UploadFile's internal pointer is at the end.

        # 2. Rewind the UploadFile stream to the beginning so it can be read again for saving.
        # FastAPI's UploadFile is typically a SpooledTemporaryFile which supports seek.
        await resume_file.seek(0)

        # 3. Save the file to disk using the (now rewound) UploadFile stream.
        # The save_upload_file function will read from the stream and then close resume_file.
        file_path_on_disk = await file_handling.save_upload_file(resume_file, file_handling.UPLOADS_DIR_RESUMES)
        logger.info(f"Resume uploaded and saved to: {file_path_on_disk}")
        # At this point, resume_file (the UploadFile object) is closed by save_upload_file.

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
        # Pass full text and optionally identified sections to NER
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


# --- Question Generation Endpoint ---

@router.post("/questions/generate/", response_model=schemas.QuestionGenerationResponse)
async def generate_interview_questions_endpoint(request: schemas.QuestionGenerationRequest):
    try:
        # Basic filtering of skills before sending to generator
        cleaned_skills = [s for s in request.extracted_skills if s and len(s) > 1]

        questions = question_generator.generate_llm_interview_questions( # Call the new main function
            resume_text=request.resume_text,
            job_description_text=request.job_description_text,
            extracted_skills=cleaned_skills, 
            role_title=request.role_title,
            num_questions=5 # Request 5 questions
        )

        if not questions or (len(questions) == 1 and "Failed to generate" in questions[0]):
            logger.warning("No meaningful questions were generated by LLM.")
            # Return the failure message if present
            return schemas.QuestionGenerationResponse(
                questions=questions, 
                message="LLM could not generate questions based on input. Check logs or try different input."
            )

        return schemas.QuestionGenerationResponse(
            questions=questions,
            message="Interview questions generated successfully using LLM."
        )
    except Exception as e:
        logger.error(f"Error in /questions/generate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during question generation: {e}")

# --- STUB Endpoints for Future Video/Audio Features ---

@router.post("/videos/upload/", response_model=schemas.VideoUploadResponse, tags=["Video (Future)"])
async def upload_video_for_analysis(video_file: UploadFile = File(...)):
    """
    (STUB) Uploads a video file for future transcription and analysis.
    """
    logger.info(f"Received video upload request (stub): {video_file.filename}")
    
    video_id_base = "placeholder_video_id"
    if video_file.filename:
        # Sanitize filename a bit for a dummy ID
        safe_filename_part = "".join(c if c.isalnum() else "_" for c in video_file.filename.split('.')[0])
        video_id_base = f"vid_{safe_filename_part}"
        
    video_id = f"{video_id_base}_{random.randint(1000,9999)}"

    # In a real implementation, you would save the video file first:
    # try:
    #     await video_file.seek(0) # Ensure stream is at the beginning if read before
    #     video_path = await file_handling.save_upload_file(video_file, file_handling.UPLOADS_DIR_VIDEOS)
    #     logger.info(f"Video '{video_file.filename}' saved to {video_path} (stub)")
    # except Exception as e:
    #     logger.error(f"Stub: Failed to save video '{video_file.filename}': {e}")
    #     raise HTTPException(status_code=500, detail=f"Stub: Error saving video: {e}")
    # finally:
    #     if not video_file.file.closed: # Ensure it's closed if save_upload_file didn't
    #         await video_file.close()


    return schemas.VideoUploadResponse(
        video_id=video_id, 
        message=f"Video '{video_file.filename}' received (stub functionality). Ready for transcription/analysis."
    )


@router.get("/videos/transcribe/{video_id}/", response_model=schemas.TranscriptionResponse, tags=["Video (Future)"])
async def transcribe_video_audio(video_id: str):
    """
    (STUB) Transcribes the audio from a previously uploaded video.
    """
    logger.info(f"Received transcription request for video_id (stub): {video_id}")
    
    # Dummy path, in real app, you'd look up based on video_id
    dummy_audio_path = f"uploads/videos/{video_id}.mp3" # Or .wav, etc.
    transcription = transcription_service.transcribe_audio_file_whisper_cpu(audio_file_path=dummy_audio_path) 
    
    return schemas.TranscriptionResponse(
        transcription=transcription,
        message=f"Transcription for video {video_id} (stub)."
    )

@router.get("/videos/analyze/{video_id}/", response_model=schemas.VideoAnalysisResponse, tags=["Video (Future)"])
async def analyze_video_performance(video_id: str):
    """
    (STUB) Performs visual analysis on a previously uploaded video.
    """
    logger.info(f"Received video analysis request for video_id (stub): {video_id}")
    
    dummy_video_path = f"uploads/videos/{video_id}.mp4" 
    analysis_result_dict = video_analysis_service.analyze_video_with_mediapipe_opencv(video_file_path=dummy_video_path)
    
    if "error" in analysis_result_dict: 
        raise HTTPException(status_code=500, detail=analysis_result_dict["error"])

    analysis_schema = schemas.VideoAnalysisResult(**analysis_result_dict)

    return schemas.VideoAnalysisResponse(
        analysis=analysis_schema,
        message=f"Video analysis for {video_id} complete (stub)."
    )

@router.get("/feedback/{session_id}/", response_model=schemas.FeedbackResponse, tags=["Video (Future)"])
async def get_overall_feedback(session_id: str): 
    """
    (STUB) Provides overall performance feedback based on transcription,
    visual analysis, and resume-role fit.
    """
    logger.info(f"Received feedback request for session_id (stub): {session_id}")

    dummy_transcription_quality = {"clarity": 0.8, "keyword_relevance": 0.7}
    dummy_visual_analysis = {"eye_contact_score": 0.65, "engagement_level": "Fair"}
    dummy_resume_role_alignment = {"match_score": 0.7, "missing_skills": ["kubernetes"]}

    feedback_dict = feedback_service.generate_performance_feedback(
        transcription_quality=dummy_transcription_quality,
        visual_analysis=dummy_visual_analysis,
        resume_role_alignment=dummy_resume_role_alignment
    )
    
    return schemas.FeedbackResponse(
        overall_score=feedback_dict.get("overall_score", 0.0),
        strengths=feedback_dict.get("strengths", []),
        areas_for_improvement=feedback_dict.get("areas_for_improvement", []),
        message=f"Overall feedback for session {session_id} (stub)."
    )
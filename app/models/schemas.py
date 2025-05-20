from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class BaseResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None

class ExtractedEntities(BaseModel):
    skills: List[str] = Field(default_factory=list)
    experience_sections: List[str] = Field(default_factory=list) # Simplified for now
    education: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)

class ResumeParseResponse(BaseResponse):
    filename: str
    text_content: str
    entities: Optional[ExtractedEntities] = None

class JobDescriptionInput(BaseModel):
    job_description_text: str

class ResumeDataInput(BaseModel):
    resume_text: str
    resume_entities: ExtractedEntities

class ComparisonResult(BaseModel):
    tfidf_similarity_score: Optional[float] = None
    semantic_similarity_score: Optional[float] = None
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills_from_jd: List[str] = Field(default_factory=list)
    # resume_skills_not_in_jd: List[str] = Field(default_factory=list)

class CompareWithJDRequest(BaseModel):
    resume_text: str # Full text for TF-IDF/Semantic on whole doc
    extracted_skills: List[str] # Specific skills for direct comparison
    job_description_text: str

class CompareWithJDResponse(BaseResponse):
    comparison: Optional[ComparisonResult] = None

class QuestionGenerationRequest(BaseModel):
    resume_text: str
    job_description_text: str
    extracted_skills: List[str]
    # Optional: role, company, etc. for more tailored questions
    role_title: Optional[str] = None

class QuestionGenerationResponse(BaseResponse):
    questions: List[str] = Field(default_factory=list)

# For future video upload
class VideoUploadResponse(BaseResponse):
    video_id: str
    message: str

class TranscriptionResponse(BaseResponse):
    transcription: str

class VideoAnalysisResult(BaseModel):
    eye_contact_score: float
    facial_expression_summary: str
    engagement_level: str

class VideoAnalysisResponse(BaseResponse):
    analysis: Optional[VideoAnalysisResult] = None

class FeedbackResponse(BaseResponse):
    overall_score: float
    strengths: List[str]
    areas_for_improvement: List[str]
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware 
from pathlib import Path
import logging
import os
from dotenv import load_dotenv

from .models import schemas
from .services import (
    resume_parser,
    entity_extractor,
    role_matcher,
    question_generator,
    transcription_service, 
    video_analysis_service, 
    feedback_service 
)
from .utils import file_handling
from .api import routes as api_routes # Import the routes

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Application Setup ---
description = """
InterviewAI Studio API helps users prepare for job interviews.
It allows resume parsing, skill extraction, job description matching,
and interview question generation.
Future features include video interview simulation and analysis. 🚀
"""

app = FastAPI(
    title="InterviewAI Studio",
    description=description,
    version="0.1.0",
    license_info={ # Optional
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)



# --- Include Routers ---
# This is where you'd include routers from other files if you split them further.
# For this project, we'll define routes directly or use the one from api_routes.py
app.include_router(api_routes.router, prefix="/api/v1", tags=["Core Features"])


# --- Root Endpoint (Optional - for health check or basic info) ---
@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Welcome to InterviewAI Studio API!",
        "documentation_swagger": "/docs",
        "documentation_redoc": "/redoc"
        }

# --- Lifecycle Events  ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing resources...")
    file_handling.ensure_upload_dirs_exist()
    logger.info("Attempting to pre-load NER model...")
    entity_extractor.get_ner_pipeline()
    logger.info("Attempting to pre-load Sentence Transformer model...")
    role_matcher.get_sentence_transformer_model()
    # logger.info("Attempting to pre-load T5 Question Generation model...") 
    # question_generator.preload_llm_models() # New
    # logger.info("Model pre-loading attempts complete.")
    # logger.info("Application startup complete.")

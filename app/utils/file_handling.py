import os
import shutil
from fastapi import UploadFile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent # interviewai/
UPLOADS_DIR_RESUMES = BASE_DIR / "uploads" / "resumes"
UPLOADS_DIR_VIDEOS = BASE_DIR / "uploads" / "videos" # For future use

def ensure_upload_dirs_exist():
    UPLOADS_DIR_RESUMES.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR_VIDEOS.mkdir(parents=True, exist_ok=True)

async def save_upload_file(upload_file: UploadFile, destination_folder: Path) -> Path:
    ensure_upload_dirs_exist()
    try:
        file_path = destination_folder / upload_file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return file_path
    finally:
        await upload_file.close()

def get_resume_path(filename: str) -> Path:
    return UPLOADS_DIR_RESUMES / filename

def get_video_path(filename: str) -> Path: # For future use
    return UPLOADS_DIR_VIDEOS / filename

# Call this once at startup or ensure it's called before any save operation
ensure_upload_dirs_exist()
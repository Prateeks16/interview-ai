import fitz  
from docx import Document
from io import BytesIO
from typing import Tuple, Optional
import re

def parse_pdf_to_text(file_bytes: bytes) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        raise ValueError(f"Could not parse PDF: {e}")
    return text

def parse_docx_to_text(file_bytes: bytes) -> str:
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error parsing DOCX: {e}")
        raise ValueError(f"Could not parse DOCX: {e}")
    return text

def extract_text_from_resume(filename: str, file_bytes: bytes) -> Tuple[str, Optional[str]]:
    
    if filename.lower().endswith(".pdf"):
        try:
            return parse_pdf_to_text(file_bytes), None
        except ValueError as e:
            return "", str(e)
    elif filename.lower().endswith(".docx"):
        try:
            return parse_docx_to_text(file_bytes), None
        except ValueError as e:
            return "", str(e)
    else:
        return "",

# --- Basic Section Extraction ---
# This can be improved significantly with more robust regex or NLP techniques
def extract_sections_from_text(text: str) -> dict:
    sections = {}
    # Keywords that might indicate the start of a section (case-insensitive)
    # Order matters if sections can be nested or have ambiguous headers
    section_keywords = {
        "experience": r"(?i)(?:work|professional)\s*experience|employment\s*history|career\s*summary",
        "education": r"(?i)education|academic\s*background",
        "skills": r"(?i)skills|technical\s*skills|proficiencies",
        "projects": r"(?i)projects|personal\s*projects|portfolio",
        "summary": r"(?i)summary|objective|profile",
    }

    # Find all keyword matches and their positions
    found_keywords = []
    for key, pattern in section_keywords.items():
        for match in re.finditer(pattern, text):
            found_keywords.append({"key": key, "start": match.start(), "header": match.group(0)})

    # Sort keywords by their start position
    found_keywords.sort(key=lambda x: x["start"])

    if not found_keywords:
        # If no keywords found, treat the whole text as 'general' or 'summary'
        # Or handle as an error/unstructured. For now, just return empty.
        return {"unstructured_text": text}


    # Iterate through sorted keywords to define section boundaries
    for i, current_kw in enumerate(found_keywords):
        start_pos = current_kw["start"]
        # End position is the start of the next keyword, or end of text
        end_pos = found_keywords[i+1]["start"] if i + 1 < len(found_keywords) else len(text)
        
        # Extract the text for the current section (from end of its header to start of next header)
        # This tries to get content *after* the header.
        header_end_pos = start_pos + len(current_kw["header"])
        section_content = text[header_end_pos:end_pos].strip()
        
        # If multiple "experience" sections are found, append them or handle as needed
        # For simplicity, this will overwrite if the same key is found (e.g. two "SKILLS" headers)
        # A better approach might be to collect all content under similar keys.
        if current_kw["key"] in sections and section_content:
             sections[current_kw["key"]] += "\n" + section_content
        elif section_content:
            sections[current_kw["key"]] = section_content
            
    # If there's text before the first recognized section header, capture it.
    if found_keywords and found_keywords[0]["start"] > 0:
        pre_section_text = text[:found_keywords[0]["start"]].strip()
        if pre_section_text:
            sections["introduction_or_contact"] = pre_section_text
            
    # If no sections were populated but text exists, put all text in 'unstructured_text'
    if not sections and text:
        sections["unstructured_text"] = text.strip()

    return sections
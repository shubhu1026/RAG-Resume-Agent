import re
import fitz
import os
from docx import Document

def anonymize_resume(text: str) -> str:
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'EMAIL_REDACTED', text
    )
    text = re.sub(
        r'(\+?\d{1,3}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)\d{3}\s*[\-.\s]\s*\d{4}',
        'PHONE_REDACTED', text
    )
    text = re.sub(r'https?://\S+', 'URL_REDACTED', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text

def extract_text_from_file(file_obj):
    # Get the file path (Gradio gives a temp file path)
    file_path = file_obj.name

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text

    elif ext == ".docx":
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")
import PyPDF2
import re

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
    return text

def extract_text_from_pdf(file_path: str) -> str:
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join(page.extract_text() for page in reader.pages)

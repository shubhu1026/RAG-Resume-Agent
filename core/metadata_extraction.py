from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List, Union, Optional
from langchain_openai import ChatOpenAI
from app.config import MODEL_NAME

class ResumeMetadata(BaseModel):
    skills: Optional[List[str]] = None
    project_headers: Optional[List[str]] = None
    tech_stack: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    experience_headers: Optional[List[str]] = None
    education_headers: Optional[List[Union[str, dict]]] = None
    section_headers: Optional[List[str]] = None  

def extract_metadata(resume_text: str) -> ResumeMetadata:
    parser = PydanticOutputParser(pydantic_object=ResumeMetadata)
    llm = ChatOpenAI(model_name=MODEL_NAME)
    response = llm.invoke(
        f"""
        Extract metadata from the resume text in the given format.
        
        Resume text:
        {resume_text}
        
        {parser.get_format_instructions()}
        """
    )

    if hasattr(response, "content"):  # AIMessage object
        response = response.content
    elif isinstance(response, list):  # list of AIMessage
        response = " ".join([r.content if hasattr(r, "content") else str(r) for r in response])
    
    return parser.parse(response)

def convert_metadata_for_chroma(metadata: ResumeMetadata) -> dict:
    def list_to_str(lst):
        if not lst: return ""
        out = []
        for item in lst:
            if isinstance(item, dict):
                degree = item.get("degree", "")
                institution = item.get("institution", "")
                duration = item.get("duration", "")
                out.append(f"{degree} at {institution} ({duration})".strip())
            else:
                out.append(str(item))
        return ", ".join(out)

    return {
        "skills": list_to_str(metadata.skills),
        "projects": list_to_str(metadata.project_headers),
        "tech_stack": list_to_str(metadata.tech_stack),
        "tags": list_to_str(metadata.tags),
        "experience": list_to_str(metadata.experience_headers) or "",
        "education": list_to_str(metadata.education_headers),
        "section_headers": list_to_str(metadata.section_headers),
    }

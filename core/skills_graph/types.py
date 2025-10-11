from typing_extensions import TypedDict
from pydantic import BaseModel
from typing import List

class SkillsState(TypedDict):
    resume_text: str
    job_description: str
    skills_comparison: dict  
    
class SkillsComparison(BaseModel):
    resume_skills: List[str]
    job_description_skills: List[str]
    matching_skills: List[str]
    missing_skills: List[str]
    ats_recommendations: List[str]
    fit_score: float
    fit_reasoning: str

class ResumeSuggestions(BaseModel):
    suggestions: List[str]
    sample_bullets: List[str]
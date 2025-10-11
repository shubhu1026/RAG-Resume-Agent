from typing_extensions import TypedDict
from typing import List
from pydantic import BaseModel, Field

class GraphState(TypedDict):
    question: str
    generation: str
    resume_text: str
    documents: List[str]
    job_description: str 
    chat_history: List[dict]

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="yes/no if it answers the question")
    hallucination: str = Field(description="yes/no if contains unsupported info")

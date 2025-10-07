from langgraph.graph import END, StateGraph, START 
from typing_extensions import TypedDict
from app.config import MODEL_NAME, TEMPERATURE
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_models import ChatOpenAI
import json
from pydantic import BaseModel
from typing import List
import re

class SuggestionsState(TypedDict):
    resume_text: str
    job_description: str
    skills_comparison: dict
    suggestions: str

class SkillsComparison(BaseModel):
    matched_skills: List[str]
    missing_skills: List[str]
    extra_skills: List[str]

skill_comp_parser = PydanticOutputParser(pydantic_object=SkillsComparison)

def extract_and_compare(state):
    """
    state: dict containing "resume_text" and "job_description"
    """
    resume_text = state["resume_text"]
    jd_text = state["job_description"]

    prompt_text = f"""
    You are an expert career coach.

    1. Extract a list of skills from the resume below.
    2. Extract a list of skills from the job description below.
    3. Return ONLY JSON following this schema:
    {skill_comp_parser.get_format_instructions()}

    Resume:
    {resume_text}

    Job Description:
    {jd_text}
    """

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    response = llm.predict(prompt_text)
    print("LLM response:", repr(response))

    # Parse directly into Pydantic model
    skills_comparison = skill_comp_parser.parse(response)
    return skills_comparison

class ResumeSuggestions(BaseModel):
    suggestions: List[str]
    sample_bullets: List[str]

suggestions_parser = PydanticOutputParser(pydantic_object=ResumeSuggestions)

def generate_resume_suggestions(resume_text, jd_text, skills_comparison):
    prompt = f"""
You are a career coach.

Resume Skills Comparison:
{skills_comparison}

Provide:
1. 3 concrete suggestions to improve the resume.
2. 2 sample bullet points with measurable impact.

Respond ONLY in JSON following this schema:
{suggestions_parser.get_format_instructions()}
"""
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    response = llm.predict(prompt)
    print("LLM response:", repr(response))

    # Parse into Pydantic model safely
    return suggestions_parser.parse(response)

def build_suggestions_flow():
    workflow = StateGraph(SuggestionsState)

    def run_suggestions(state):
        # your extraction + suggestion code here
        skills_comparison = extract_and_compare(state)
        suggestions = generate_resume_suggestions(state["resume_text"], state["job_description"], skills_comparison)
        return {
            "resume_text": state["resume_text"],
            "job_description": state["job_description"],
            "skills_comparison": skills_comparison,
            "suggestions": suggestions
        }

    workflow.add_node("generate_suggestions", run_suggestions)

    # 🚨 Connect START to your first node
    workflow.add_edge(START, "generate_suggestions")
    workflow.add_edge("generate_suggestions", END)

    return workflow.compile()


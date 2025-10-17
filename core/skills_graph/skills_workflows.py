from langgraph.graph import END, StateGraph, START 
from app.config import MODEL_NAME, TEMPERATURE
from langchain.output_parsers import PydanticOutputParser
from core.skills_graph.types import ResumeSuggestions, SkillsComparison, SkillsState
from core.prompts import extract_skills_runnable

skill_comp_parser = PydanticOutputParser(pydantic_object=SkillsComparison)

suggestions_parser = PydanticOutputParser(pydantic_object=ResumeSuggestions)

# --- Functions ---
def extract_and_compare(state):
    response = extract_skills_runnable.invoke({
        "resume_text": state["resume_text"],
        "jd_text": state["job_description"],
        "format_instructions": skill_comp_parser.get_format_instructions()
    })

    # extract string from AIMessage
    if hasattr(response, "content"):
        response_text = response.content
    else:
        response_text = response

    try:
        return skill_comp_parser.parse(response_text)
    except Exception as e:
        print("Error parsing skills:", e)
        return SkillsComparison(matched_skills=[], missing_skills=[], extra_skills=[])

# --- Workflow ---
def build_skills_flow():
    """
    StateGraph for skills comparison, optionally generating suggestions
    """
    workflow = StateGraph(SkillsState)

    def run_skills(state):
        skills_comparison = extract_and_compare(state)
        output = {
            "resume_text": state["resume_text"],
            "job_description": state["job_description"],
            "skills_comparison": skills_comparison
        }

        return output

    workflow.add_node("compute_skills", run_skills)
    workflow.add_edge(START, "compute_skills")
    workflow.add_edge("compute_skills", END)

    return workflow.compile()

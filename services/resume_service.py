import gradio as gr
from core.resume_processing import extract_text_from_file, anonymize_resume
from core.metadata_extraction import extract_metadata, convert_metadata_for_chroma
from services.summarize_service import summarize_job_description
from core.vectorstore import create_vectorstore
from core.graph.workflow import build_workflow
from langchain_community.tools.tavily_search import TavilySearchResults

from app.config import MODEL_NAME, TEMPERATURE
from langchain_openai import ChatOpenAI

def process_resume_file(file_obj, job_description, app_state=None):
    if file_obj is None:
        return "⚠️ Please upload a resume file.", None, None, ""

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

    # Extract and anonymize resume
    resume_text = extract_text_from_file(file_obj)
    safe_text = anonymize_resume(resume_text)

    # Extract metadata
    metadata = extract_metadata(safe_text)
    metadata_dict = convert_metadata_for_chroma(metadata)

    # Build workflow
    vector_db = create_vectorstore(safe_text, metadata_dict)
    web_search_tool = TavilySearchResults()
    workflow = build_workflow(vector_db, web_search_tool)

    # Summarize JD if present
    jd_summary = summarize_job_description(job_description, llm) if job_description else ""

    # Update app_state as a dictionary
    if app_state is None or not isinstance(app_state, dict):
        app_state = {}

    app_state.update({
        "workflow": workflow,          # preserve workflow for chat
        "resume_text": safe_text,      # for skills comparison
        "job_description": jd_summary  # optional summary
    })

    return "✅ Resume processed!", app_state, metadata_dict, jd_summary


def process_job_desc(job_description, app_state, metadata_state):
    if not app_state:
        return "⚠️ Please upload a resume first.", app_state, metadata_state, ""

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    jd_summary = summarize_job_description(job_description, llm) if job_description else ""

    return "✅ Job description updated!", app_state, metadata_state, jd_summary

from core.prompts import summarize_jd_runnable

def summarize_job_description(job_description: str, llm):
    if not job_description or not job_description.strip():
        return ""
    
    summary = summarize_jd_runnable.invoke({"job_description": job_description})
    return summary
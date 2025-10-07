
def auto_resume_fn(app_state, metadata_state, jd_state):
    if not app_state:
        return "⚠️ Please upload a resume first.", None
    
    # Generate tailored resume using adaptive RAG
    state = {
        "resume_text": str(metadata_state),
        "job_description": jd_state or "",
        "question": "Rewrite resume sections to better match the job description"
    }
    
    generated_resume = ""
    for output in app_state.stream(state):
        for _, value in output.items():
            if "generation" in value:
                generated_resume = value["generation"]
    
    # Save to temporary file for download
    file_path = "tailored_resume.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(generated_resume)
    
    return "✅ Tailored Resume Generated!", file_path
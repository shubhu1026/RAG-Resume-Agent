from core.suggestions_graph import build_suggestions_flow

def skills_fit_fn(app_state, metadata_state, jd_state):
    if not app_state:
        return "⚠️ Please upload a resume first.", {}
    
    # Use the workflow to extract skills and compare with JD
    suggestions_workflow = build_suggestions_flow()
    state = {
        "resume_text": str(metadata_state),
        "job_description": jd_state or "",
    }
    result = suggestions_workflow.invoke(state)
    
    fit_score = result.get("fit_score", "N/A")
    matching_skills = result.get("skills_comparison", [])
    missing_skills = result.get("missing_skills", [])
    
    output_json = {
        "Fit Score": fit_score,
        "Matching Skills": matching_skills,
        "Missing Skills": missing_skills
    }
    return "✅ Skills & Fit Analysis Complete!", output_json
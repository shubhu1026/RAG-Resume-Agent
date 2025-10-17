from core.skills_graph.skills_workflows import build_skills_flow
from core.skills_graph.skills_workflows import SkillsComparison 
from langchain.output_parsers import PydanticOutputParser

def skills_fit_fn(state, metadata_state, jd_state):
    """Analyze resume vs job description and return skill comparison with ATS suggestions."""

    resume_str = state.get("resume_text", "")
    workflow = state.get("workflow")  # keep workflow for other logic if needed

    if not resume_str:
        return SkillsComparison(
            resume_skills=[],
            job_description_skills=[],
            matching_skills=[],
            missing_skills=[],
            ats_recommendations=[]
        )

    # Build skills workflow
    skills_workflow = build_skills_flow(generate_suggestions=True)
    result = skills_workflow.invoke({
        "resume_text": resume_str,
        "job_description": jd_state or ""
    })

    skills_data = result.get("skills_comparison", result)

    if isinstance(skills_data, SkillsComparison):
        return skills_data
    elif isinstance(skills_data, dict):
        return SkillsComparison(**skills_data)
    else:
        raise TypeError(f"Unexpected type for skills_data: {type(skills_data)}")


def render_skill_chips(state, metadata_state, jd_state):
    """Render color-coded chips for skills and return suggestions + fit score."""
    skills_comparison = skills_fit_fn(state, metadata_state, jd_state)

    matched = skills_comparison.matching_skills
    missing = skills_comparison.missing_skills
    ats_recs = skills_comparison.ats_recommendations

    html = "<div style='display:flex; flex-wrap:wrap; gap:0.5rem;'>"
    for skill in matched:
        html += f"<span style='background:#4ade80; color:#fff; padding:0.3rem 0.6rem; border-radius:12px;' title='Matching Skill'>{skill}</span>"
    for skill in missing:
        html += f"<span style='background:#facc15; color:#000; padding:0.3rem 0.6rem; border-radius:12px;' title='Missing Skill'>{skill}</span>"
    html += "</div>"

    fit_summary = f"✅ Fit Score: {skills_comparison.fit_score:.1f}/100\nReasoning: {skills_comparison.fit_reasoning}"

    return fit_summary, html, ats_recs

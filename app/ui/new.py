import gradio as gr
from core.resume_processing import extract_text_from_file, anonymize_resume
from metadata_extraction import extract_metadata, convert_metadata_for_chroma
from suggestions_graph import build_suggestions_flow
from vectorstore import create_vectorstore
from graph.graph import build_workflow
from langchain_community.tools.tavily_search import TavilySearchResults

from app.config import LANGCHAIN_PROJECT, MODEL_NAME, TEMPERATURE

from langsmith import Client
from langsmith.run_helpers import traceable

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

# --- New helper function to run suggestions flow ---
def run_suggestions_flow(resume_text, jd_text):
    suggestions_workflow = build_suggestions_flow()
    state = {"resume_text": resume_text, "job_description": jd_text}
    result = suggestions_workflow.invoke(state)
    # return both skills comparison and AI suggestions text
    skills_text = f"Matched Skills: {result['skills_comparison'].matched_skills}\n" \
              f"Missing Skills: {result['skills_comparison'].missing_skills}\n" \
              f"Extra Skills: {result['skills_comparison'].extra_skills}"

    suggestions_text = "\n".join(result['suggestions'].suggestions) + "\n\n" + \
                       "\n".join(result['suggestions'].sample_bullets)
    return skills_text, suggestions_text

@traceable(name="RAG-Chat-Interaction")
def chat_fn(message, history, app_state, metadata_state, jd_state):
    if not app_state:
        return "⚠️ Please upload a resume first."

    inputs = {
        "question": message,
        "metadata_summary": str(metadata_state) if metadata_state else "",
        "job_description": jd_state or "",
    }

    final_answer = "⚠️ No response generated."
    for output in app_state.stream(inputs):
        for _, value in output.items():
            if "generation" in value:
                final_answer = value["generation"]

    return final_answer

# --- Gradio UI ---
with gr.Blocks(title="RAG Resume Agent") as demo:
    gr.Markdown("## 📄 RAG Resume Assistant\nUpload your resume, optionally add a job description, see personalized suggestions, and chat!")

    with gr.Row():
        resume_file = gr.File(label="Upload Resume (PDF/Word)", file_types=[".pdf", ".docx"])
        job_desc = gr.Textbox(label="Job Description (Optional)", lines=5,
                              placeholder="Paste job description here...")

    status = gr.Textbox(label="Status", interactive=False)

    # States
    app_state = gr.State(None)
    metadata_state = gr.State(None)
    jd_state = gr.State("")
    resume_text_state = gr.State("")  # store resume text for suggestions
    jd_text_state = gr.State("")      # store JD text for suggestions

    # Process resume file
    def process_resume(file_obj, jd_text):
        if not file_obj:
            return "⚠️ Please upload a resume file.", None, None, "", ""
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
        resume_text = extract_text_from_file(file_obj)
        safe_text = anonymize_resume(resume_text)
        metadata = extract_metadata(safe_text)
        metadata_dict = convert_metadata_for_chroma(metadata)
        vector_db = create_vectorstore(safe_text, metadata_dict)
        web_search_tool = TavilySearchResults()
        app = build_workflow(vector_db, web_search_tool)
        return "✅ Resume processed!", app, metadata_dict, safe_text, jd_text or ""

    resume_file.upload(
        fn=process_resume,
        inputs=[resume_file, job_desc],
        outputs=[status, app_state, metadata_state, resume_text_state, jd_text_state]
    )

    job_desc.change(
        fn=lambda jd, app, metadata, r_text: ("✅ JD updated!", app, metadata, r_text, jd),
        inputs=[job_desc, app_state, metadata_state, resume_text_state],
        outputs=[status, app_state, metadata_state, resume_text_state, jd_text_state]
    )

    # --- Suggestions Section ---
    with gr.Row():
        skills_box = gr.Textbox(label="Skills Analysis", lines=6, interactive=False)
        suggestions_box = gr.Textbox(label="AI Suggestions", lines=8, interactive=False)

    analyze_btn = gr.Button("Generate Suggestions")
    analyze_btn.click(
        fn=run_suggestions_flow,
        inputs=[resume_text_state, jd_text_state],
        outputs=[skills_box, suggestions_box]
    )

    # --- Chatbot Section ---
    chatbot = gr.ChatInterface(
        fn=lambda msg, hist, app, meta, jd: chat_fn(msg, hist, app, meta, jd),
        type='messages',
        additional_inputs=[app_state, metadata_state, jd_text_state]
    )

if __name__ == "__main__":
    demo.launch()

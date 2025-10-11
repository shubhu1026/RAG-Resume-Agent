import gradio as gr
from services.chat_service import chat_stream
from services.resume_service import process_resume_file, process_job_desc
from services.skills_service import render_skill_chips
from app.config import LANGCHAIN_PROJECT
from langsmith import Client

client = Client()
print(f"✅ LangSmith connected. Project: {LANGCHAIN_PROJECT}")

# --- Workflow functions ---
def process_resume_job(resume_file, job_desc_text):
    status, app_state, metadata_state, jd_state = process_resume_file(resume_file, job_desc_text)
    if job_desc_text.strip():
        status, app_state, metadata_state, jd_state = process_job_desc(job_desc_text, app_state, metadata_state)

    fit_summary, skills_html, ats_recs = render_skill_chips(app_state, metadata_state, jd_state)
    return status, app_state, metadata_state, jd_state, fit_summary, skills_html, ats_recs


def update_resume_job(resume_file, job_desc_text, app_state, metadata_state, jd_state):
    resume_file_to_use = resume_file if resume_file is not None else app_state
    jd_text_to_use = job_desc_text if job_desc_text.strip() else jd_state
    return process_resume_job(resume_file_to_use, jd_text_to_use)

# --- Theme & CSS ---
custom_theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    radius_size="lg",
    font=["Inter", "sans-serif"]
).set(
    body_background_fill="#f7f6fb",
    block_background_fill="#ffffffcc",
    block_border_width="1px",
    block_border_color="#e6e6e6",
    block_shadow="0 4px 12px rgba(0,0,0,0.05)",
    button_primary_background_fill="linear-gradient(135deg, #7f5af0 0%, #6246ea 100%)",
    button_primary_text_color="white",
    input_background_fill="#f9f9fb",
    input_border_color="#d8d8e3"
)

css = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
.gr-row, .gr-column { flex-direction: column !important; }
input[type="file"], textarea, input, .gr-button { width: 100% !important; box-sizing: border-box; }
.skill-chip-legend { display:flex; gap:1rem; margin-top:0.5rem; }
.skill-chip-legend span { padding:0.3rem 0.6rem; border-radius:12px; font-size:0.9rem; }
"""

with gr.Blocks(theme=custom_theme, css=css, title="Adaptive RAG Resume Agent") as demo:

    # States
    app_state = gr.State(None)
    metadata_state = gr.State(None)
    jd_state = gr.State("")

    # Header
    gr.HTML("""<div style="text-align:center; padding:2rem;">
        <h1 style='font-size:2rem; font-weight:700; background:linear-gradient(90deg,#7f5af0,#6246ea); 
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>📄 Adaptive RAG Resume Assistant</h1>
        <p style='color:#555; margin-top:0.5rem;'>Upload your resume and optionally a job description. Press Process to view skills and chat insights.</p>
    </div>""")

    # --- Upload Section ---
    with gr.Column(visible=True) as upload_section:
        resume_file = gr.File(label="Upload Resume (PDF/DOCX)", file_types=[".pdf", ".docx"])
        job_desc = gr.Textbox(label="Job Description (Optional)", lines=5, placeholder="Paste job description here...")
        process_btn = gr.Button("Process Resume & Job Description", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)

    # --- Results Section ---
    with gr.Column(visible=False) as results_section:
        # Skills comparison + ATS suggestions
        with gr.Row():
            with gr.Column(scale=3):
                fit_summary_display = gr.Textbox(label="Fit Summary", interactive=False)
                skills_html_display = gr.HTML(label="Skills Comparison")
                gr.HTML("""<div class="skill-chip-legend">
                    <span style="background:#4ade80; color:#fff;">Matching Skill</span>
                    <span style="background:#facc15; color:#000;">Missing Skill</span>
                </div>""")
                ats_output_display = gr.JSON(label="ATS Recommendations")
            with gr.Column(scale=1):
                gr.Markdown("### Update Resume / JD")
                update_resume_file = gr.File(label="Upload New Resume", file_types=[".pdf", ".docx"])
                update_job_desc = gr.Textbox(label="Update Job Description", lines=5)
                update_btn = gr.Button("Reprocess Updated Inputs", variant="secondary")

        # Chatbot
        gr.Markdown("### 💬 Chatbot Q&A")
        chatbot_display = gr.Chatbot(label="RAG Chatbot", height=400, type="messages")
        user_input = gr.Textbox(label="Ask a question", placeholder="Type here...")
        send_btn = gr.Button("Send", variant="primary")

    # --- Process Action ---
    def process_action(resume_file, job_desc_text):
        status_text, app_s, meta_s, jd_s, fit_summary, skills_html, ats_recs = process_resume_job(resume_file, job_desc_text)
        return (
            gr.update(visible=False),  # hide upload section
            gr.update(visible=True),   # show results section
            status_text, app_s, meta_s, jd_s, fit_summary, skills_html, ats_recs
        )

    process_btn.click(
        process_action,
        inputs=[resume_file, job_desc],
        outputs=[upload_section, results_section, status, app_state, metadata_state, jd_state,
                 fit_summary_display, skills_html_display, ats_output_display]
    )

    # --- Update Action ---
    update_btn.click(
        lambda new_resume, new_jd, app_s, meta_s, jd_s: update_resume_job(new_resume, new_jd, app_s, meta_s, jd_s),
        inputs=[update_resume_file, update_job_desc, app_state, metadata_state, jd_state],
        outputs=[status, app_state, metadata_state, jd_state,
                 fit_summary_display, skills_html_display, ats_output_display]
    )

    # --- Chatbot ---
    def submit_stream(message, app_s, meta_s, jd_s):

        # Initialize app_state if None
        if app_s is None:
            app_s = {}
        if "chat_history" not in app_s or app_s["chat_history"] is None:
            app_s["chat_history"] = []

        history = app_s["chat_history"]

        # Append user message once
        user_msg = {"role": "user", "content": message}
        history.append(user_msg)
        yield history, app_s

        # Append assistant placeholder
        assistant_msg = {"role": "assistant", "content": ""}
        history.append(assistant_msg)
        assistant_idx = len(history) - 1
        yield history, app_s

        workflow_state = {
            "question": message,
            "metadata_summary": str(meta_s) if meta_s else "",
            "job_description": jd_s or app_s.get("job_description", ""),
            "chat_history": history[-8:]  # last N messages
        }

        workflow = app_s.get("workflow")
        if not workflow:
            print("Workflow not initialized!")
            history[assistant_idx]["content"] = "⚠️ Workflow not initialized."
            yield history, app_s
            return

        final_text = ""
        for output in workflow.stream(workflow_state):
            for _, value in output.items():
                if "generation" in value:
                    token = getattr(value["generation"], "content", str(value["generation"]))
                    final_text += token
                    history[assistant_idx]["content"] = final_text
                    yield history, app_s

        history[assistant_idx]["content"] = final_text
        yield history, app_s

    user_input.submit(
        chat_stream,
        inputs=[user_input, chatbot_display, app_state, metadata_state, jd_state],
        outputs=[chatbot_display],
        queue=True
    )
    send_btn.click(
        chat_stream,
        inputs=[user_input, chatbot_display, app_state, metadata_state, jd_state],
        outputs=[chatbot_display],
        queue=True
    )


if __name__ == "__main__":
    demo.launch()

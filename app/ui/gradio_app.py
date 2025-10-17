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
.upload-fixed {
    max-width: 45%; /* upload section stays narrow */
    margin: auto;
}

.results-fullwidth { 
    max-width: 100% !important; /* results section stretches full width */
    margin: 0 !important;
}

input[type="file"], textarea, input, .gr-button { 
    width: 100% !important; 
    box-sizing: border-box; 
}
.skill-chip-legend { 
    display:flex; gap:1rem; margin-top:0.5rem; 
}
.skill-chip-legend span { 
    padding:0.3rem 0.6rem; border-radius:12px; font-size:0.9rem; 
}

"""

with gr.Blocks(theme=custom_theme, css=css, title="Adaptive RAG Resume Agent") as demo:

    # States
    app_state = gr.State(None)
    metadata_state = gr.State(None)
    jd_state = gr.State("")

    # Header
    gr.HTML("""<div style="text-align:center; padding:2rem;">
        <h1 style='font-size:2rem; font-weight:700; background:linear-gradient(90deg,#7f5af0,#6246ea); 
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>Adaptive RAG Resume Assistant</h1>
        <p style='color:#555; margin-top:0.5rem;'>Upload your resume and optionally a job description. Press Process to view skills and chat insights.</p>
    </div>""")

    # --- Upload Section ---
    with gr.Column(visible=True, elem_classes="upload-fixed") as upload_section:
        resume_file = gr.File(label="Upload Resume (PDF/DOCX)", file_types=[".pdf", ".docx"])
        job_desc = gr.Textbox(label="Job Description (Optional)", lines=5, placeholder="Paste job description here...")
        process_btn = gr.Button("Process Resume & Job Description", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
    
    # --- Results Section (3-column layout) ---
    with gr.Column(visible=False,  elem_classes="results-fullwidth") as results_section:
        with gr.Row():
            gr.Markdown("")
            update_btn = gr.Button("🔄 Update Resume / JD", variant="primary", scale=0.25)

        with gr.Row(elem_classes="gr-row"):
            # Left column: Skills
            with gr.Column(scale=1, visible=False) as skills_section:
                gr.Markdown("### Skills Comparison")
                fit_summary_display = gr.Textbox(label="Fit Summary", interactive=False)
                skills_html_display = gr.HTML(label="Skills Comparison")
                gr.HTML("""<div class="skill-chip-legend">
                    <span style="background:#4ade80; color:#fff;">Matching Skill</span>
                    <span style="background:#facc15; color:#000;">Missing Skill</span>
                </div>""")

            # Middle column: Chatbot
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chatbot Q&A")
                chatbot_display = gr.Chatbot(label="RAG Chatbot", height=400, type="messages")
                user_input = gr.Textbox(label="Ask a question", placeholder="Type here...")
                send_btn = gr.Button("Send", variant="primary")

            # Right column: ATS + Update Resume / JD
            with gr.Column(scale=1, visible=False) as ats_section:
                gr.Markdown("### ATS Recommendations & Update Inputs")
                ats_output_display = gr.JSON(label="ATS Recommendations")          

    # --- Process Action ---
    def process_action(resume_file, job_desc_text):
        status_text, app_s, meta_s, jd_s, fit_summary, skills_html, ats_recs = process_resume_job(resume_file, job_desc_text)
        show_sections = bool(job_desc_text.strip())

        return (
            gr.update(visible=False),   # Hide upload section
            gr.update(visible=True),    # Show results section
            gr.update(visible=show_sections),  # Skills section
            gr.update(visible=show_sections),  # ATS section
            status_text, app_s, meta_s, jd_s, fit_summary, skills_html, ats_recs
        )


    process_btn.click(
        process_action,
        inputs=[resume_file, job_desc],
        outputs=[
            upload_section, results_section, skills_section, ats_section, 
            status, app_state, metadata_state, jd_state,
            fit_summary_display, skills_html_display, ats_output_display
        ]
    )


    # --- Update Action ---
    def update_action():
        return (
            gr.update(visible=True),     # Show upload section
            gr.update(visible=False),    # Hide results section
            gr.update(value=""),         # Clear status
            None, None, "",              # Reset states
            gr.update(value=[]),         # Clear chatbot
            gr.update(value=""),         # Clear fit summary
            gr.update(value=""),         # Clear skills HTML
            gr.update(value={})          # Clear ATS JSON
        )


    update_btn.click(
        update_action,
        inputs=[],
        outputs=[
            upload_section, results_section, status,
            app_state, metadata_state, jd_state,
            chatbot_display, fit_summary_display,
            skills_html_display, ats_output_display
        ]
    )

    # --- Chatbot ---
    user_input.submit(
        chat_stream,
        inputs=[user_input, chatbot_display, app_state, metadata_state, jd_state, user_input, send_btn],
        outputs=[chatbot_display, user_input, send_btn],
        queue=True
    )

    send_btn.click(
        chat_stream,
        inputs=[user_input, chatbot_display, app_state, metadata_state, jd_state, user_input, send_btn],
        outputs=[chatbot_display, user_input, send_btn],
        queue=True
    )


if __name__ == "__main__":
    demo.launch()

import gradio as gr
from services.chat_service import chat_fn
from services.resume_service import process_job_desc, process_resume_file
from services.resume_tailor_service import auto_resume_fn
from services.skills_service import skills_fit_fn
from skills_graph.skills_workflows import build_skills_flow

from app.config import LANGCHAIN_PROJECT

from langsmith import Client

client = Client()
print(f"✅ LangSmith connected. Project: {LANGCHAIN_PROJECT}")

# def on_resume_upload(resume_text, jd_text):
#     suggestions_workflow = build_suggestions_flow()
#     state = {"resume_text": resume_text, "job_description": jd_text}
#     result = suggestions_workflow.invoke(state)
#     return result["skills_comparison"], result["suggestions"]

def on_resume_upload(resume_text, jd_text):
    suggestions_workflow = build_skills_flow(generate_suggestions=False)
    state = {"resume_text": resume_text, "job_description": jd_text}
    result = suggestions_workflow.invoke(state)
    # 'suggestions' key may not exist if generate_suggestions=False
    return result["skills_comparison"], result.get("suggestions", {})


# --- Theme ---
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

# --- CSS for stable layout ---
css = """
.gradio-container {
    max-width: 850px !important;
    margin: auto !important;
}
body {
    background: linear-gradient(180deg,#f7f6fb 0%,#f0eef8 100%) !important;
    overflow-x: hidden !important;
}
.gr-row, .gr-column {
    flex-direction: column !important;
}
input[type="file"], textarea, input, .gr-button {
    width: 100% !important;
    box-sizing: border-box;
}
.full-width-input .gr-file-input,
#resume-upload .gr-file-input {
    width: 100% !important;
    min-width: 850px !important;
    box-sizing: border-box;
}
.full-width-input .gr-file-input span {
    max-width: calc(100% - 50px);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: inline-block;
    vertical-align: middle;
}
"""

# --- Build UI ---
with gr.Blocks(theme=custom_theme, css=css, title="Adaptive RAG Resume Agent") as demo:

    # --- Header ---
    gr.HTML("""
    <div style="text-align:center; padding:2rem;">
        <h1 style='font-size:2rem; font-weight:700; background:linear-gradient(90deg,#7f5af0,#6246ea); 
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            📄 Adaptive RAG Resume Assistant
        </h1>
        <p style='color:#555; margin-top:0.5rem;'>
            Upload your resume, optionally add a job description, and explore AI-powered insights!
        </p>
    </div>
    """)

    # --- Upload Section ---
    gr.Markdown("### 📤 Upload Inputs")
    resume_file = gr.File(
        label="Upload Resume (PDF/DOCX)",
        file_types=[".pdf", ".docx"],
        elem_id="resume-upload",
        elem_classes="full-width-input"
    )
    job_desc = gr.Textbox(
        label="Job Description (Optional)",
        lines=5,
        placeholder="Paste job description here..."
    )
    status = gr.Textbox(label="Status", interactive=False, placeholder="System messages appear here...")

    # --- States ---
    app_state = gr.State(None)
    metadata_state = gr.State(None)
    jd_state = gr.State("")

    # --- Bind uploads ---
    resume_file.upload(
        fn=process_resume_file,
        inputs=[resume_file, job_desc],
        outputs=[status, app_state, metadata_state, jd_state]
    )
    job_desc.change(
        fn=process_job_desc,
        inputs=[job_desc, app_state, metadata_state],
        outputs=[status, app_state, metadata_state, jd_state]
    )

    # --- Tabs ---
    with gr.Tabs():

        # --- Chatbot Q&A ---
        with gr.Tab("💬 Chatbot Q&A"):
            gr.Markdown("""
            Ask questions about your resume or career path. The AI provides multi-step reasoning responses.
            """)
            chatbot_display = gr.Chatbot(label="RAG Chatbot", height=450, type="messages", elem_id="chatbox")
            user_input = gr.Textbox(label="Ask a question", placeholder="Type here...", interactive=True)
            send_btn = gr.Button("Send", variant="primary")  # <-- new button

            def submit(message, history, app_state, metadata_state, jd_state):
                # Use our updated chat_fn with static context + truncated recent messages
                history = chat_fn(message, history, app_state, metadata_state, jd_state)
                return history, history

            # Chatbot tab
            user_input.submit(
                submit,
                inputs=[user_input, chatbot_display, app_state, metadata_state, jd_state],
                outputs=[chatbot_display, chatbot_display]
            )
            send_btn.click(
                submit,
                inputs=[user_input, chatbot_display, app_state, metadata_state, jd_state],
                outputs=[chatbot_display, chatbot_display]
            )



        # --- Skills & Job Fit ---
        # with gr.Tab("📊 Skills & Job Fit"):
        #     fit_status = gr.Textbox(label="Status", interactive=False)
        #     fit_output = gr.JSON(label="Fit Analysis")
        #     gr.Button("Analyze Skills & Fit", variant="primary").click(
        #         skills_fit_fn,
        #         inputs=[app_state, metadata_state, jd_state],
        #         outputs=[fit_status, fit_output]
        #     )

        # --- Skills & Job Fit Tab ---
        with gr.Tab("📊 Skills & Job Fit"):
            fit_status = gr.Textbox(label="Status", interactive=False)
            skill_chips = gr.HTML(label="Skill Comparison")
            
            def render_skill_chips(state, metadata_state, jd_state):
                """
                Build colored HTML chips for matched, missing, extra skills
                """
                # Call your skills_fit_fn to get JSON output
                skills_comparison = skills_fit_fn(state, metadata_state, jd_state)
                
                matched = skills_comparison["matched_skills"]
                missing = skills_comparison["missing_skills"]
                extra = skills_comparison["extra_skills"]

                # Build HTML for chips
                html = "<div style='display:flex; flex-wrap:wrap; gap:0.5rem;'>"
                for skill in matched:
                    html += f"<span style='background:#4ade80; color:#fff; padding:0.3rem 0.6rem; border-radius:12px;' title='Matched Skill'>{skill}</span>"
                for skill in missing:
                    html += f"<span style='background:#facc15; color:#000; padding:0.3rem 0.6rem; border-radius:12px;' title='Missing Skill'>{skill}</span>"
                for skill in extra:
                    html += f"<span style='background:#f87171; color:#fff; padding:0.3rem 0.6rem; border-radius:12px;' title='Extra Skill'>{skill}</span>"
                html += "</div>"

                return "Skills comparison generated.", html

            analyze_btn = gr.Button("Analyze Skills & Fit", variant="primary")
            analyze_btn.click(
                render_skill_chips,
                inputs=[app_state, metadata_state, jd_state],
                outputs=[fit_status, skill_chips]
            )

            # Optional: Button to generate suggestions later
            gen_sugg_btn = gr.Button("Generate Suggestions", variant="secondary")
            gen_sugg_output = gr.JSON(label="Suggestions Output")
            # gen_sugg_btn.click(fn=generate_suggestions_fn, inputs=[...], outputs=[gen_sugg_output])


        # --- Auto-Tailored Resume ---
        with gr.Tab("📝 Auto-Tailored Resume"):
            resume_status = gr.Textbox(label="Status", interactive=False)
            resume_file_output = gr.File(label="Download Tailored Resume")
            gr.Button("Generate Tailored Resume", variant="primary").click(
                auto_resume_fn,
                inputs=[app_state, metadata_state, jd_state],
                outputs=[resume_status, resume_file_output]
            )

    # --- Footer ---
    gr.HTML("""
    <div style='text-align:center; padding:1rem; color:#777; font-size:0.9rem'>
        Adaptive RAG Resume Assistant
    </div>
    """)

if __name__ == "__main__":
    demo.launch()

import gradio as gr
from resume_processing import extract_text_from_pdf, anonymize_resume
from metadata_extraction import extract_metadata, convert_metadata_for_chroma
from vectorstore import create_vectorstore
from graph import build_workflow
from langchain_community.tools.tavily_search import TavilySearchResults

from config import LANGCHAIN_PROJECT, MODEL_NAME, TEMPERATURE

from langsmith import Client
from langsmith.run_helpers import traceable

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

client = Client()
print(f"✅ LangSmith connected. Project: {LANGCHAIN_PROJECT}")

# --- Job Description Summarizer ---
def summarize_job_description(job_description: str, llm):
    if not job_description or not job_description.strip():
        return ""

    summarize_prompt = PromptTemplate(
        input_variables=["job_description"],
        template="""
You are an assistant that processes job descriptions. 
Keep all important and relevant information (skills, technologies, responsibilities, qualifications, and role-specific details). 
Remove anything unnecessary, repetitive, or generic.

Here is the job description:
{job_description}

Return a clean, concise summary that captures only the essential information for understanding the role and matching a resume.
"""
    )
    chain = LLMChain(llm=llm, prompt=summarize_prompt)
    return chain.run(job_description).strip()


# --- Resume + JD Processing ---
def process_resume(file_obj, job_description):
    """Processes resume and JD automatically when file uploaded."""
    if file_obj is None:
        return "⚠️ Please upload a resume file.", None, None, ""
    
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

    # 1. Extract and anonymize resume
    resume_text = extract_text_from_pdf(file_obj.name)
    safe_text = anonymize_resume(resume_text)

    # 2. Extract metadata
    metadata = extract_metadata(safe_text)
    metadata_dict = convert_metadata_for_chroma(metadata)

    # 3. Build vector DB + app
    vector_db = create_vectorstore(safe_text, metadata_dict)
    web_search_tool = TavilySearchResults()
    app = build_workflow(vector_db, web_search_tool)

    # 4. Summarize JD
    jd_summary = summarize_job_description(job_description, llm)

    return "✅ Resume processed! You can now chat with the agent.", app, metadata_dict, jd_summary


@traceable(name="RAG-Chat-Interaction")
def chat_fn(message, history, app_state, metadata_state, jd_state):
    """Handles chatbot interactions using states stored in gr.State"""
    if app_state is None:
        return "⚠️ Please upload a resume first."

    inputs = {
        "question": message,
        "metadata_summary": str(metadata_state),
        "job_description": jd_state,
    }

    final_answer = "⚠️ No response generated."
    for output in app_state.stream(inputs):
        for _, value in output.items():
            if "generation" in value:
                final_answer = value["generation"]

    return final_answer


# --- Gradio UI ---
with gr.Blocks(title="RAG Resume Agent") as demo:
    gr.Markdown("## 📄 RAG Resume Assistant\nUpload your resume, add an optional job description, then chat!")

    with gr.Row():
        resume_file = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
        job_desc = gr.Textbox(label="Job Description (Optional)", lines=5,
                              placeholder="Paste job description here...")

    status = gr.Textbox(label="Status", interactive=False)

    # states
    app_state = gr.State(None)
    metadata_state = gr.State(None)
    jd_state = gr.State("")

    # Process automatically on file upload or JD change
    resume_file.upload(
        fn=process_resume,
        inputs=[resume_file, job_desc],
        outputs=[status, app_state, metadata_state, jd_state]
    )
    job_desc.change(
        fn=process_resume,
        inputs=[resume_file, job_desc],
        outputs=[status, app_state, metadata_state, jd_state]
    )

    # Chatbot 
    chatbot = gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(label="RAG Agent"),
        additional_inputs=[app_state, metadata_state, jd_state]
    )

if __name__ == "__main__":
    demo.launch()

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

client = Client()
print(f"✅ LangSmith connected. Project: {LANGCHAIN_PROJECT}")

# --- Job Description Summarizer ---
def summarize_job_description(job_description: str, llm):
    if not job_description or not job_description.strip():
        return ""

    
    chain = LLMChain(llm=llm, prompt=summarize_prompt)
    return chain.run(job_description).strip()


# --- Resume Processing Only ---
# def process_resume_file(file_obj, job_description):
#     if file_obj is None:
#         return "⚠️ Please upload a resume file.", None, None, ""
    
#     llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

#     app_state = None
#     metadata_state = None
#     jd_state = ""

#     # 1. Extract and anonymize resume
#     resume_text = extract_text_from_file(file_obj)
#     safe_text = anonymize_resume(resume_text)
#     with open("cleaned_text.txt", "w", encoding="utf-8") as f:
#         f.write(safe_text)

#     # 2. Extract metadata
#     metadata = extract_metadata(safe_text)
#     metadata_dict = convert_metadata_for_chroma(metadata)

#     # 3. Build vector DB + app
#     vector_db = create_vectorstore(safe_text, metadata_dict)
#     web_search_tool = TavilySearchResults()
#     app = build_workflow(vector_db, web_search_tool)

#     # 4. Summarize JD (if present)
#     jd_summary = summarize_job_description(job_description, llm) if job_description else ""

#     return "✅ Resume processed!", app, metadata_dict, jd_summary

def process_resume_file(file_obj, job_description):
    if file_obj is None:
        return "⚠️ Please upload a resume file.", None, None, ""

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

    # 1. Extract and anonymize resume
    resume_text = extract_text_from_file(file_obj)
    safe_text = anonymize_resume(resume_text)

    # 2. Extract metadata
    metadata = extract_metadata(safe_text)
    metadata_dict = convert_metadata_for_chroma(metadata)

    # 3. Build a *fresh* vector DB + workflow
    vector_db = create_vectorstore(safe_text, metadata_dict)
    web_search_tool = TavilySearchResults()
    app = build_workflow(vector_db, web_search_tool)

    # 4. Summarize JD (if present)
    jd_summary = summarize_job_description(job_description, llm) if job_description else ""

    # 🚨 Return fresh state objects so old ones don’t persist
    return "✅ Resume processed!", app, metadata_dict, jd_summary


def process_job_desc(job_description, app_state, metadata_state):
    if not app_state:
        return "⚠️ Please upload a resume first.", app_state, metadata_state, ""

    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    jd_summary = summarize_job_description(job_description, llm) if job_description else ""

    return "✅ Job description updated!", app_state, metadata_state, jd_summary

def on_resume_upload(resume_text, jd_text):
    suggestions_workflow = build_suggestions_flow()
    state = {"resume_text": resume_text, "job_description": jd_text}
    result = suggestions_workflow.run(state)
    return result["skills_comparison"], result["suggestions"]

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
    gr.Markdown("## 📄 RAG Resume Assistant\nUpload your resume, add an optional job description, then chat!")

    with gr.Row():
        resume_file = gr.File(label="Upload Resume (PDF)", file_types=[".pdf", ".docx"])
        job_desc = gr.Textbox(label="Job Description (Optional)", lines=5,
                              placeholder="Paste job description here...")

    status = gr.Textbox(label="Status", interactive=False)

    # states
    app_state = gr.State(None)
    metadata_state = gr.State(None)
    jd_state = gr.State("")

    # Process automatically on file upload or JD change
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


    # Chatbot 
    chatbot = gr.ChatInterface(
        fn=chat_fn,
        type='messages',
        # chatbot=gr.Chatbot(label="RAG Agent"),
        additional_inputs=[app_state, metadata_state, jd_state]
    )

if __name__ == "__main__":
    demo.launch()

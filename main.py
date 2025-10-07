from core.resume_processing import extract_text_from_pdf, anonymize_resume
from metadata_extraction import extract_metadata, convert_metadata_for_chroma
from vectorstore import create_vectorstore
from rag_chain import build_rag_chain
from utils import pretty_print
from graph.graph import build_workflow
from langchain_community.tools.tavily_search import TavilySearchResults

def main():
    # Step 1: Load + anonymize resume
    resume_text = extract_text_from_pdf("SUPER RESUME.pdf")
    safe_text = anonymize_resume(resume_text)

    # Step 2: Metadata
    metadata = extract_metadata(safe_text)
    metadata_dict = convert_metadata_for_chroma(metadata)

    # Step 3: Vectorstore
    vector_db = create_vectorstore(safe_text, metadata_dict)

    # Step 4: Build workflow
    web_search_tool = TavilySearchResults()
    app = build_workflow(vector_db, web_search_tool)

    # Step 5: Example query
    inputs = {
        "question": "What changes can I make to my resume to better fit the job description?",
        "metadata_summary": str(metadata_dict),
        "job_description": "AI/ML Engineer role requiring cloud security, governance, etc."
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            pretty_print(f"Node {key}", value)

    print("\n✅ Final Answer:", value.get("generation", ""))

if __name__ == "__main__":
    main()

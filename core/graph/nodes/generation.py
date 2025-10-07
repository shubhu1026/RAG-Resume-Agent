# core/rag_chain.py (updated)
from core.prompts import rag_runnable, compressor_runnable
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def generate(state, vector_db, rag_chain=None):
    """
    Generate an answer using RAG with optional compressed retriever.
    """
    query = state["question"]
    jd = getattr(state.get("job_description", ""), "content", state.get("job_description", ""))
    jd = jd.strip() if jd else ""

    # Compressed retriever setup
    try:
        base_retriever = vector_db.as_retriever(search_kwargs={"k": 6})
        compressed_retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            compressor=LLMChainExtractor(llm_chain=compressor_runnable)
        )
    except Exception:
        compressed_retriever = base_retriever

    # Fetch relevant documents
    context_docs = compressed_retriever.invoke(query) if compressed_retriever else vector_db.similarity_search(query, k=3)
    context_text = "\n\n".join([d.page_content for d in context_docs])

    # Prepare input for RAG runnable
    llm_input = f"Job Description:\n{jd}\n\nResume Info:\n{context_text}" if jd else f"Resume Info:\n{context_text}"

    # Use RunnableSequence instead of direct LLM call
    response = rag_runnable.invoke({"context": context_text, "question": query})

    return {"question": query, "generation": response, "documents": context_docs}


def llm_fallback(state):
    """
    Simple fallback for small talk or uncategorized queries.
    """
    question = state["question"]
    # Can create a mini RunnableSequence if you want, or just call the RAG prompt
    fallback_prompt = {"context": "", "question": f"Answer the question/ greeting/ small talk concisely: {question}"}
    generation = rag_runnable.invoke(fallback_prompt)
    return {"question": question, "generation": generation}

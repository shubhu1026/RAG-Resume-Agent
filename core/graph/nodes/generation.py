from core.prompts import rag_runnable, llm 

def generate(state, rag_chain=None):
    """
    Generate a professional, accurate answer using RAG.
    """
    query = state.get("question", "").strip()
    jd = state.get("job_description", "")
    jd_text = getattr(jd, "content", jd).strip() if jd else ""

    if not query:
        return {"generation": "⚠️ No query provided.", "documents": []}

    # --- Use documents from previous retrieval if available ---
    context_docs = state.get("documents", [])
    if not context_docs:
        return {"generation": "⚠️ No context documents available.", "documents": []}

    context_text = "\n\n".join([d.page_content for d in context_docs])

    prompt_inputs = {
        "context": context_text,
        "job_description": jd_text,
        "question": query
    }

    try:
        response = rag_runnable.invoke(prompt_inputs)
        final_answer = getattr(response, "content", str(response)).strip()
    except Exception as e:
        print(f"RAG generation failed: {e}")
        final_answer = "⚠️ Unable to generate a response at this time."

    return {
        "question": query,
        "generation": final_answer,
        "documents": context_docs
    }

def llm_fallback(state):
    """
    Fallback LLM response if RAG fails or no context.
    """
    question = state.get("question", "")
    job_description = state.get("job_description", "")

    fallback_prompt = {
        "context": "",
        "job_description": job_description or "",
        "question": (
            f"The user said: '{question}'. "
            "If this is a greeting or small talk, respond politely and professionally in 1–2 sentences. "
            "If it’s a general career or job-related question, provide a short, insightful response."
        )
    }

    generation = rag_runnable.invoke(fallback_prompt)
    return {"question": question, "generation": generation}

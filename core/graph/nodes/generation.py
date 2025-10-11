from core.prompts import rag_runnable, llm_fallback_runnable

def format_history_for_prompt(history):
    """Format structured chat history into text for the prompt."""
    formatted = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"
    return formatted.strip()


def generate(state, rag_chain=None):
    """Main generation node for RAG-based responses (no app_state)."""
    query = state.get("question", "").strip()
    jd = state.get("job_description", "")
    jd_text = getattr(jd, "content", jd).strip() if jd else ""

    if not query:
        return {"generation": "⚠️ No query provided.", "documents": []}

    context_docs = state.get("documents", [])
    if not context_docs:
        return {"generation": "⚠️ No context documents available.", "documents": []}

    context_text = "\n\n".join([d.page_content for d in context_docs])
    chat_history = state.get("chat_history", [])
    conversation_text = format_history_for_prompt(chat_history)

    prompt_inputs = {
        "context": context_text,
        "job_description": jd_text,
        "question": query,
        "conversation_history": conversation_text,
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
        "documents": context_docs,
    }


def llm_fallback(state):
    """Fallback when RAG fails or retrieval gives no context."""
    question = state.get("question", "").strip()
    if not question:
        return {"generation": "⚠️ No question provided."}

    chat_history = state.get("chat_history", [])
    conversation_text = format_history_for_prompt(chat_history)

    try:
        prompt_inputs = {"question": question, "conversation_history": conversation_text}
        generation = llm_fallback_runnable.invoke(prompt_inputs)
        final_answer = getattr(generation, "content", str(generation)).strip()
    except Exception as e:
        print(f"LLM fallback failed: {e}")
        final_answer = "⚠️ Unable to generate a response at this time."

    return {"question": question, "generation": final_answer}

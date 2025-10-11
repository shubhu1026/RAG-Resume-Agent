# core/routing_fn.py
from core.routing import should_use_vectorstore
from core.prompts import routing_runnable

def format_history_for_routing(history):
    if not history:
        return ""
    formatted = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"
    return formatted

def route_question(state, vector_db):
    """
    Decide how to route a user's question: vectorstore, web search, or LLM fallback.
    """
    question = state["question"]
    q_lower = question.lower()
    jd_summary = getattr(state.get("job_description", ""), "content", state.get("job_description", ""))
    jd_summary = jd_summary.lower() if jd_summary else ""

    # Step 1: Quick keyword rule
    if jd_summary and any(word in q_lower for word in ["company", "role", "position", "salary", "location"]):
        return "retrieve"

    # Step 2: Embedding similarity heuristic
    try:
        if should_use_vectorstore(question, vector_db, threshold=0.70, k=3):
            return "retrieve"
    except Exception:
        pass

    # Step 3: LLM-based routing via RunnableSequence
    route_msg = routing_runnable.invoke({
        "user_question": question,
        "metadata_summary": state.get("metadata_summary", ""),
        "jd_summary": state.get("job_description", ""),
        "conversation_history": format_history_for_routing(state.get("chat_history", [])),
    })

    route = route_msg.content.strip().lower()

    print("ROUTE:", route)

    if route == "vectorstore":
        return "retrieve"
    elif route == "websearch":
        return "web_search"
    else:
        return "llm_fallback"

from langsmith.run_helpers import traceable

@traceable(name="RAG-Chat-Interaction")
def chat_fn(message, history, app_state, metadata_state, jd_state, max_recent=4):
    """
    Handle chat messages using static context from resume + JD (app_state + metadata_state),
    and only pass recent user messages to the model to reduce token usage.
    """
    if not app_state or "workflow" not in app_state:
        return [{"role": "assistant", "content": "⚠️ Please upload a resume first."}]

    # Keep only the last N messages for context
    recent_history = (history or [])[-max_recent*2:]  # history is list of dicts

    # Prepare inputs for RAG workflow
    inputs = {
        "question": message,
        "metadata_summary": str(metadata_state) if metadata_state else "",
        "job_description": jd_state or app_state.get("job_description", ""),
        "recent_conversation": recent_history
    }

    final_answer = "⚠️ No response generated."
    workflow = app_state["workflow"]  

    for output in workflow.stream(inputs):
        for _, value in output.items():
            if "generation" in value:
                gen = value["generation"]
                # Extract text if it's an AIMessage from RunnableSequence
                final_answer = getattr(gen, "content", str(gen))

    # Initialize history if None
    history = history or []

    # Append user and assistant messages as dicts (Gradio-compatible)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final_answer})

    return history

import time
from langsmith.run_helpers import traceable
import gradio as gr

@traceable(name="RAG-Chat-Interaction")
def chat_fn(message, history, app_state, metadata_state, jd_state):
    """Non-streaming fallback chat function."""
    history = history or []
    if not app_state or "workflow" not in app_state:
        return [{"role": "assistant", "content": "⚠️ Please upload a resume first."}]

    inputs = {
        "question": message,
        "metadata_summary": str(metadata_state) if metadata_state else "",
        "job_description": jd_state or app_state.get("job_description", ""),
        "documents": app_state.get("documents", []),
        "chat_history": history[-8:],  # recent history
    }

    workflow = app_state["workflow"]
    final_answer = "⚠️ No response generated."

    for output in workflow.stream(inputs):
        for _, value in output.items():
            if "generation" in value:
                gen = value["generation"]
                final_answer = getattr(gen, "content", str(gen))

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final_answer})
    return history

def chat_stream(message, history, app_state, metadata_state, jd_state, user_input, send_btn):
    history = history or []

    # Step 1: Append user message and disable input + button
    user_msg = {"role": "user", "content": message}
    history.append(user_msg)
    app_state.setdefault("chat_history", []).append(user_msg)
    yield history, gr.update(interactive=False), gr.update(interactive=False, value="Send")

    # Step 2: Assistant placeholder
    assistant_msg = {"role": "assistant", "content": "generating response..."}
    history.append(assistant_msg)
    yield history, None, gr.update(interactive=False, value="Send")

    # Step 3: Stream workflow response
    workflow_state = {
        "question": message,    
        "metadata_summary": str(metadata_state or ""),
        "job_description": jd_state or app_state.get("job_description", ""),
        "documents": app_state.get("documents", []),
        "chat_history": app_state["chat_history"],
        "app_state": app_state,
    }

    final_text = ""
    workflow = app_state.get("workflow")
    if not workflow:
        history[-1]["content"] = "⚠️ Workflow not initialized."
        yield history, None, gr.update(interactive=False, value="Send")
        return

    for output in workflow.stream(workflow_state):
        for _, value in output.items():
            if "generation" in value:
                token = getattr(value["generation"], "content", str(value["generation"]))
                final_text += token
                history[-1]["content"] = final_text
                yield history, None, gr.update(interactive=False, value="Send")
                time.sleep(0.02)

    # Step 4: finalize assistant response and re-enable input + button
    history[-1]["content"] = final_text
    app_state["chat_history"].append({"role": "assistant", "content": final_text})
    yield history, gr.update(interactive=True), gr.update(interactive=True, value="Send")

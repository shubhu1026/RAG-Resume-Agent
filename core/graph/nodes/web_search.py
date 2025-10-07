from core.prompts import web_search_runnable

def web_search(state, web_search_tool):
    """
    Perform a web search and generate a concise answer using LLM.
    Returns a dict with 'question' and 'generation'.
    """
    question = state.get("question", "")
    if not question:
        return {"question": "", "generation": "⚠️ No question provided for web search."}

    # Get web search results
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d.get("content", "") for d in docs])

    # Use RunnableSequence to generate answer
    generation = web_search_runnable.invoke({"question": question, "web_results": web_results})

    return {"question": question, "generation": getattr(generation, "content", str(generation))}
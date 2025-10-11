

# def retrieve(state):
    #     question = state["question"]

    #     # Expand query into multiple perspectives
    #     expanded_queries = expand_query(question, num_variations=5)

    #     # Retrieve docs for each perspective
    #     retrieved_docs = []
    #     for q in expanded_queries:
    #         docs = vector_db.similarity_search(q, k=3)
    #         retrieved_docs.extend(docs)

    #     # Remove duplicate documents
    #     unique_docs = {d.page_content: d for d in retrieved_docs}.values()

    #     return {"question": question, "documents": list(unique_docs)}

# core/query_expansion.py
from core.prompts import expand_query_runnable  # this is your RunnableSequence for query expansion

def expand_query(user_query: str, num_variations: int = 5):
    """
    Expand a user query into multiple perspectives using RunnableSequence.
    """
    # Invoke the RunnableSequence instead of ChatOpenAI directly
    response = expand_query_runnable.invoke({"user_query": user_query, "num_variations": num_variations})

    text = getattr(response, "content", str(response))

    variations = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            parts = line.split(". ", 1)
            variations.append(parts[1].strip() if len(parts) == 2 else line)

    return [v for v in variations if v]

def retrieve(state, vector_db):
    """
    Retrieve documents from vector_db based on the user's base query
    and expanded query perspectives.
    """
    question = state["question"]

    try:
        base_retriever = vector_db.as_retriever(search_kwargs={"k": 6})
    except Exception:
        base_retriever = None

    # Expand query using RunnableSequence
    expanded_queries = expand_query(question, num_variations=3)

    # Include the original query as well
    all_queries = [question] + expanded_queries

    retrieved_docs = []

    if base_retriever:
        for q in all_queries:
            retrieved_docs.extend(base_retriever.get_relevant_documents(q))
    else:
        for q in all_queries:
            retrieved_docs.extend(vector_db.similarity_search(q, k=3))

    # Deduplicate documents
    unique_docs = {d.page_content: d for d in retrieved_docs}.values()

    return {"question": question, "documents": list(unique_docs)}

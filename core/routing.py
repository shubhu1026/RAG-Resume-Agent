from langchain_openai import OpenAIEmbeddings

def should_use_vectorstore(question: str, vector_db, threshold: float = 0.70, k: int = 3):
    """
    Quick heuristic: embed the question and check similarity against the top resume documents.
    If any top doc has cosine similarity > threshold -> prefer VECTORSTORE.
    Returns True if VECTORSTORE preferred, False otherwise.
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        q_emb = embeddings.embed_query(question)
        docs = vector_db.similarity_search_by_vector(q_emb, k=k, include_metadata=True)
        # Chroma doc similarity score access can differ; attempt to read similarity from metadata or fallback
        for d in docs:
            # doc.metadata might contain a '_similarity' key depending on impl
            score = d.metadata.get("_similarity", None) if hasattr(d, "metadata") else None
            # if no explicit score, we can compute approximate by re-embedding and cosine (expensive)
            if score is not None:
                if float(score) >= threshold:
                    return True
        # fallback: return False (conservative)
        return False
    except Exception:
        return False

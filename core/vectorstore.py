# vectorstore.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import COLLECTION_NAME
from core.resume_chunking import chunk_resume
import os

DEFAULT_PERSIST_DIR = os.getenv("VECTORSTORE_PERSIST_DIR", "./resume_db")

def save_docs_to_file(docs, filename="chunked_resume.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs, 1):
            f.write("="*60 + "\n")
            f.write(f"📄 Document {i}\n")
            f.write(f"🔖 Metadata: {doc.metadata}\n")
            f.write("📝 Content:\n")
            f.write(doc.page_content.strip() + "\n")
            f.write("="*60 + "\n\n")
    print(f"Saved {len(docs)} documents to {filename}")

def _fallback_text_split(resume_text: str, metadata: dict):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_text(resume_text)
    documents = [Document(page_content=chunk, metadata={**metadata, "type":"chunk"}) for chunk in chunks]
    return documents

def create_vectorstore(resume_text: str, metadata: dict, persist_directory: str = DEFAULT_PERSIST_DIR):
    """
    Creates vectorstore using chunk_resume then fallback to character splitter.
    Returns a Chroma vectorstore (persisted).
    """
    documents = chunk_resume(resume_text, metadata)

    if not documents:
        print("chunk_resume returned no docs — using fallback splitter")
        documents = _fallback_text_split(resume_text, metadata)

    save_docs_to_file(documents, "my_chunked_resume.txt")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Ensure persist dir exists
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        collection_name=COLLECTION_NAME or "resume",
        persist_directory=persist_directory
    )

    return vectorstore

def build_vectorstore_and_retriever(resume_text: str, metadata: dict, persist_directory: str = DEFAULT_PERSIST_DIR, search_k: int = 6):
    """
    Convenience: create vectorstore and return a retriever configured with search kwargs.
    """
    vs = create_vectorstore(resume_text, metadata, persist_directory=persist_directory)
    retriever = vs.as_retriever(search_kwargs={"k": search_k})
    return vs, retriever

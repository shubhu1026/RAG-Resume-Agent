# vectorstore.py
import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import COLLECTION_NAME
from core.resume_chunking import chunk_resume

# Default directories
DEFAULT_PERSIST_DIR = os.getenv("VECTORSTORE_PERSIST_DIR", "./resume_db")
VECTORSTORE_DIR = "vectorstore/"

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def update_vectorstore(resume_text=None, job_desc_text=None, persist_directory=VECTORSTORE_DIR):
    """
    Completely removes old Chroma vectorstore and rebuilds it
    with the new resume and/or job description documents.
    """
    # 1️⃣ Delete old vectorstore safely
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
    os.makedirs(persist_directory, exist_ok=True)

    # 2️⃣ Prepare text documents
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    if resume_text:
        resume_chunks = splitter.split_text(resume_text)
        docs.extend([
            Document(page_content=chunk, metadata={"source": "resume"})
            for chunk in resume_chunks
        ])

    if job_desc_text:
        jd_chunks = splitter.split_text(job_desc_text)
        docs.extend([
            Document(page_content=chunk, metadata={"source": "job_description"})
            for chunk in jd_chunks
        ])

    if not docs:
        raise ValueError("No text provided to rebuild vectorstore.")

    save_docs_to_file(docs, filename="updated_docs.txt")

    # 3️⃣ Recreate the vectorstore
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME or "resume"
    )

    # 4️⃣ Persist and confirm
    vectorstore.persist()
    print(f"✅ Vectorstore rebuilt successfully with {len(docs)} documents!")
    return vectorstore

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
    Creates vectorstore using chunk_resume, falls back to character splitter if needed.
    Returns a persisted Chroma vectorstore.
    """
    documents = chunk_resume(resume_text, metadata)
    if not documents:
        print("chunk_resume returned no docs — using fallback splitter")
        documents = _fallback_text_split(resume_text, metadata)
    save_docs_to_file(documents, "my_chunked_resume.txt")

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
    Convenience function: creates vectorstore and returns a retriever with search kwargs.
    """
    vs = create_vectorstore(resume_text, metadata, persist_directory=persist_directory)
    retriever = vs.as_retriever(search_kwargs={"k": search_k})
    return vs, retriever

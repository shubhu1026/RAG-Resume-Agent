# vectorstore.py
import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import COLLECTION_NAME
from core.resume_chunking import chunk_resume
import uuid

# Default directories
DEFAULT_PERSIST_DIR = os.getenv("VECTORSTORE_PERSIST_DIR", "./resume_db")
VECTORSTORE_DIR = "vectorstore/"

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
    Creates a fresh vectorstore using chunk_resume, falls back to character splitter if needed.
    Ensures previous vectorstore is completely removed.
    Returns a persisted Chroma vectorstore.
    """
    # 0️⃣ Clear old vectorstore
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)
    os.makedirs(persist_directory, exist_ok=True)

    # 1️⃣ Chunk resume
    documents = chunk_resume(resume_text, metadata)
    if not documents:
        print("chunk_resume returned no docs — using fallback splitter")
        documents = _fallback_text_split(resume_text, metadata)

    # 2️⃣ Save chunks to file
    save_docs_to_file(documents, "my_chunked_resume.txt")

    # 3️⃣ Use a unique collection name to prevent merging
    collection_name = f"resume_{uuid.uuid4().hex}"

    # 4️⃣ Create new vectorstore
    vectorstore = Chroma.from_documents(
        documents,
        embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    # 5️⃣ Persist and confirm
    vectorstore.persist()
    print(f"✅ Vectorstore rebuilt successfully with {len(documents)} documents! Collection: {collection_name}")
    return vectorstore

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import COLLECTION_NAME
from resume_chunking import chunk_resume
import re

# def create_vectorstore(text: str, metadata: dict):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
#     chunks = text_splitter.split_text(text)

#     documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
#     embeddings = OpenAIEmbeddings()
#     return Chroma.from_documents(documents, embeddings, collection_name=COLLECTION_NAME)

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

def create_vectorstore(resume_text: str, metadata: dict):
    """
    Creates a vectorstore using hybrid chunking (sections + items).
    """
    documents = chunk_resume(resume_text, metadata)
    save_docs_to_file(documents, "my_chunked_resume.txt")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 
    if documents:
        vectorstore = Chroma.from_documents(documents, embeddings)
    else:
        raise ValueError("No documents to create vectorstore. Check chunking step!")
    return vectorstore
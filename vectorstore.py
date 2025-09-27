from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from config import COLLECTION_NAME

def create_vectorstore(text: str, metadata: dict):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(documents, embeddings, collection_name=COLLECTION_NAME)

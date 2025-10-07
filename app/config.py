from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Langsmith
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "RAG-Resume-Agent")
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING", "true")


# Model Settings
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0
COLLECTION_NAME = "resumes"

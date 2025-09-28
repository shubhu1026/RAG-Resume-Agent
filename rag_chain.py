from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import MODEL_NAME, TEMPERATURE

def build_rag_chain():
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    prompt_template = """
You are a helpful assistant. Use the following resume information to answer the user's question.

Resume Information:
{context}

Question:
{question}

Answer in a concise and professional manner.
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template
    )
    return LLMChain(llm=llm, prompt=prompt)

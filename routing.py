from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from config import MODEL_NAME, TEMPERATURE

def build_routing_chain():
    routing_prompt = PromptTemplate(
        input_variables=["user_question", "metadata_summary", "jd_summary"],
        template="""
You are an expert assistant who decides where to route a user's question.

The user's resume contains information about the following:
{metadata_summary}

The user also provided a job description summary:
{jd_summary}

Routing rules (priority order):
1. If the question is about the user's resume (projects, skills, experience, education) 
   or how the resume relates to the job description → respond with VECTORSTORE.
2. If the question is about very recent or ongoing current events (e.g., 2025 or later), 
   live news, or real-time trends → respond with WEBSEARCH.
3. Otherwise (including historical events, general knowledge, greetings, or casual conversation) → respond with LLM.

Respond ONLY with one of: VECTORSTORE, WEBSEARCH, or LLM.

User Question: {user_question}
"""
    )
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    return LLMChain(llm=llm, prompt=routing_prompt)

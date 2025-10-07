from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI
from app.config import MODEL_NAME, TEMPERATURE

llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

SUMMARIZE_JD_PROMPT = PromptTemplate(
        input_variables=["job_description"],
        template="""
        You are an assistant that processes job descriptions. 
        Keep all important and relevant information (skills, technologies, responsibilities, qualifications, and role-specific details). 
        Remove anything unnecessary, repetitive, or generic.

        Here is the job description:
        {job_description}

        Return a clean, concise summary that captures only the essential information for understanding the role and matching a resume.
        """
    )

ROUTING_PROMPT = PromptTemplate(
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


RAG_PROMPT = PromptTemplate(
        input_variables=["context", "question"], 
        template="""
        You are a helpful assistant. Use the following resume information to answer the user's question.

        Resume Information:
        {context}

        Question:
        {question}

        Answer in a concise and professional manner.
        """
    )

EXPAND_QUERY_PROMPT = PromptTemplate(
    input_variables=["user_query", "num_variations"],
    template="""
    Rephrase the following query into {num_variations} different ways.
    Each variation should ask the question from a slightly different perspective.

    Query: "{user_query}"

    Return as a numbered list.
    """
    )

COMPRESSOR_PROMPT = PromptTemplate(
        input_variables=["document"],
        template="Compress/Extract the essential facts from the following resume chunk; return 1-2 concise sentences.\n\n{document}"
    )

GRADE_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "generation", "context", "format_instructions"],
    template="""
    You are a grader assessing an AI answer.

    1. Check if the answer addresses/resolves the user's question.
    2. Check if the answer contains hallucinations (information not present in the context/resume provided).

    Respond ONLY in JSON using this schema:
    {format_instructions}

    User Question: {question}
    AI Answer: {generation}
    Context (resume/docs): {context}
    """
)

WEB_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question", "web_results"],
    template="""
    Question: {question}

    Context (from web search results):
    {web_results}

    Provide a concise and informative answer.
    """
)


# ---- RunnableSequences ----
summarize_jd_runnable = RunnableSequence(SUMMARIZE_JD_PROMPT | llm)
routing_runnable = RunnableSequence(ROUTING_PROMPT | llm)
rag_runnable = RunnableSequence(RAG_PROMPT | llm)
expand_query_runnable = RunnableSequence(EXPAND_QUERY_PROMPT | llm)
compressor_runnable = RunnableSequence(COMPRESSOR_PROMPT | llm)
grade_generation_runnable = RunnableSequence(GRADE_GENERATION_PROMPT | llm)
web_search_runnable = RunnableSequence(WEB_SEARCH_PROMPT | llm)

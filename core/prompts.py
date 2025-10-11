from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI
from app.config import MODEL_NAME, TEMPERATURE

llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

ROUTING_PROMPT = PromptTemplate(
    input_variables=["user_question", "metadata_summary", "jd_summary"],
    template="""
You are an intelligent routing assistant that decides which subsystem should handle the user's question.

### Priority:
Focus **primarily on the provided resume metadata**. This metadata contains crucial information about work experience, projects, skills, education, achievements, and professional context. Use it as the **first and most important reference** when deciding how to route a question.

### Context:
Resume Metadata:
{metadata_summary}

Job Description Summary:
{jd_summary}

### Decision Rules (apply in order):

1. Respond **VECTORSTORE** if the question:
   - Requires information **explicitly or implicitly contained in the resume metadata**, including:
     - Work experience, skills, education, achievements, certifications
     - Projects, initiatives, or programs listed in the metadata
     - Career alignment, growth, or suitability for a job
   - Mentions **first-person references** like "my", "I", or "user's" with metadata-relevant content.
   - **Use semantic matching**: if the user mentions or alludes to a project, skill, tool, or achievement present in the metadata, route to VECTORSTORE even if phrasing is vague.
   - Examples:
     - "Summarize my professional experience"
     - "Explain my Leafio project in detail"
     - "What are my key strengths?"
     - "Does my experience match the job?"
     - "Highlight my career summary"

2. Respond **WEBSEARCH** if the question:
   - Relates to **current events, market trends, or data after 2025**.
   - Requires **fresh or external knowledge** (e.g., salaries, emerging skills, company news).

3. Respond **LLM** if:
   - It’s a general or conversational question (greetings, generic career advice, motivation, unrelated topics).
   - It **does not require resume or job metadata context**.

**Always output only one word**: VECTORSTORE, WEBSEARCH, or LLM.

User Question:
{user_question}
"""
)





SUMMARIZE_JD_PROMPT = PromptTemplate(
    input_variables=["job_description"],
    template="""
You are a professional job description analyst specializing in talent matching and role alignment.

Your goal: create a **concise, high-fidelity summary** that retains every important element relevant to resume-job fit analysis.

Include:
- Core responsibilities and deliverables
- Key required and preferred skills (technical + soft)
- Education or certification requirements
- Tools, technologies, or frameworks
- Seniority level or leadership expectations

Do NOT include generic HR phrasing, benefits, or company fluff.

Job Description:
{job_description}

Return the summary in a clean, structured format (bullet points or short paragraphs).
"""
)



# RAG_PROMPT = PromptTemplate(
#         input_variables=["context", "question"], 
#         template="""
#         You are a helpful assistant. Use the following resume information to answer the user's question.

#         Resume Information:
#         {context}

#         Question:
#         {question}

#         Answer in a concise and professional manner.
#         """
#     )

# RAG_PROMPT = PromptTemplate(
#     input_variables=["context", "job_description", "question"],
#     template="""
# You are an **Executive Career Advisor and AI Resume Consultant**. Your responses must be **precise, factual, and strictly based on the retrieved context**.

# ### Instructions:

# 1. **Use Retrieved Context**:
#    - Base your response primarily on the content provided in retrieved_context.
#    - Only include information explicitly present in the context; do not infer or fabricate.
#    - Metadata can be referenced if helpful, but only to clarify context.

# 2. **Analyze the Job Description (if provided)**:
#    - Only use this if the user explicitly asks about job alignment, gaps, or recommendations.
#    - Provide career insights **only when relevant** to the user’s question.

# 3. **Respond Dynamically**:
#    - If the user asks about a specific **project, skill, or experience**, give a **concise, structured explanation** using the retrieved context.
#    - Do **not** provide extra recommendations, career advice, or comparisons unless explicitly requested.
#    - Keep the response professional, structured, and factual.

# 4. **Fallback**:
#    - If the context does not contain enough information to answer, respond:
#      > "The provided information is insufficient to assess this aspect accurately."

# ### Retrieved Context:
# {context}

# ### Job Description (if available):
# {job_description}

# ### User Question:
# {question}

# ### Answer:
# """
# )

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "job_description", "question"],
    template="""
You are an **Executive Career Advisor and AI Resume Consultant**. Your responses must be **precise, factual, and strictly based on the provided context**.

### Instructions:

1. **Use Retrieved Context Only**:
   - Base your response on {context}.
   - You may reason, synthesize, and compare information within the context.
   - Do **not** hallucinate or add any information not present in the context.

2. **Job Alignment (if applicable)**:
   - Use {job_description} only if the user asks about job fit, gaps, or recommendations.
   - Highlight matches, gaps, and overall fit **without external assumptions**.

3. **Answer Style**:
   - Provide concise, structured, high-level insights.
   - Avoid unnecessary advice or generalizations unless explicitly requested.

4. **Fallback**:
   - If the context is insufficient to answer, respond:
     > "The provided information is insufficient to assess this aspect accurately."

### User Question:
{question}

### Answer:
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

EXTRACT_SKILLS_PROMPT = PromptTemplate(
    input_variables=["resume_text", "jd_text", "format_instructions"],
    template="""
You are an **ATS Optimization and Career Intelligence Expert**.

Analyze both the **resume** and **job description** to produce a skills alignment report.

Tasks:
1. Extract explicit and implied skills from both sources (technical, behavioral, domain, and leadership).
2. Compare both lists:
   - **Matching Skills:** Present in both documents.
   - **Missing Skills:** Present in job description but absent in resume.
3. Provide **ATS optimization suggestions** (keyword inclusion, phrasing improvements).
4. Score the **overall job fit** from 0 to 100 based on:
   - Skill alignment
   - Experience relevance
   - Achievement quality
   - Role seniority
   - ATS readiness
5. Provide a short, data-backed reasoning for the score.

Return **only valid JSON** matching this schema:
{format_instructions}

Resume:
{resume_text}

Job Description:
{jd_text}
"""
)

RESUME_SUGGESTIONS_PROMPT = PromptTemplate(
    input_variables=["resume_text", "jd_text", "skills_comparison"],
    template="""
You are a senior resume strategist.

Using the following skills comparison:
{skills_comparison}

Provide:
1. **Three high-impact, specific resume improvement suggestions** focused on clarity, quantifiable results, and alignment with the job description.
2. **Two optimized bullet points** rewritten to show measurable outcomes and strong action verbs.

Output **only JSON** following this schema:
{format_instructions}
"""
)

# ---- RunnableSequences ----
summarize_jd_runnable = RunnableSequence(SUMMARIZE_JD_PROMPT | llm)
routing_runnable = RunnableSequence(ROUTING_PROMPT | llm)
rag_runnable = RunnableSequence(RAG_PROMPT | llm)
expand_query_runnable = RunnableSequence(EXPAND_QUERY_PROMPT | llm)
grade_generation_runnable = RunnableSequence(GRADE_GENERATION_PROMPT | llm)
web_search_runnable = RunnableSequence(WEB_SEARCH_PROMPT | llm)
extract_skills_runnable = RunnableSequence(EXTRACT_SKILLS_PROMPT | llm)
generate_suggestions_runnable = RunnableSequence(RESUME_SUGGESTIONS_PROMPT | llm)
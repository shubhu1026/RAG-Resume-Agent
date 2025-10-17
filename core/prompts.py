from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI
from app.config import MODEL_NAME, TEMPERATURE

llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

# ROUTING_PROMPT = PromptTemplate(
#     input_variables=["user_question", "metadata_summary", "jd_summary"],
#     template="""
# You are an intelligent routing assistant that decides which subsystem should handle the user's question.

# ### Priority:
# Focus **primarily on the provided resume metadata**. This metadata contains crucial information about work experience, projects, skills, education, achievements, and professional context. Use it as the **first and most important reference** when deciding how to route a question.

# ### Context:
# Resume Metadata:
# {metadata_summary}

# Job Description Summary:
# {jd_summary}

# ### Decision Rules (apply in order):

# 1. Respond **VECTORSTORE** if the question:
#    - Requires information **explicitly or implicitly contained in the resume metadata**, including:
#      - Work experience, skills, education, achievements, certifications
#      - Projects, initiatives, or programs listed in the metadata
#      - Career alignment, growth, or suitability for a job
#    - Mentions **first-person references** like "my", "I", or "user's" with metadata-relevant content.
#    - **Use semantic matching**: if the user mentions or alludes to a project, skill, tool, or achievement present in the metadata, route to VECTORSTORE even if phrasing is vague.
#    - Examples:
#      - "Summarize my professional experience"
#      - "Explain my Leafio project in detail"
#      - "What are my key strengths?"
#      - "Does my experience match the job?"
#      - "Highlight my career summary"

# 2. Respond **WEBSEARCH** if the question:
#    - Relates to **current events, market trends, or data after 2025**.
#    - Requires **fresh or external knowledge** (e.g., salaries, emerging skills, company news).

# 3. Respond **LLM** if:
#    - It’s a general or conversational question (greetings, generic career advice, motivation, unrelated topics).
#    - It **does not require resume or job metadata context**.

# **Always output only one word**: VECTORSTORE, WEBSEARCH, or LLM.

# User Question:
# {user_question}
# """
# )

ROUTING_PROMPT = PromptTemplate(
    input_variables=["user_question", "metadata_summary", "jd_summary", "conversation_history"],
    template="""
You are an intelligent **Routing Assistant** that decides which subsystem should handle the user's message.
Your task is to select one route: **VECTORSTORE**, **WEBSEARCH**, or **LLM**.

---

### 🔍 Primary Context
**Resume Metadata (most important):**
{metadata_summary}

**Job Description Summary:**
{jd_summary}

**Conversation History (for continuity):**
{conversation_history}

---

### 🧩 Routing Logic (apply in order)

1. **VECTORSTORE**
   Choose this if the question:
   - Involves details found in the **resume metadata** — including experience, projects, education, certifications, skills, or achievements.
   - Asks about **job alignment, gaps, or suitability**.
   - References previous conversation turns about resume-related topics.
   - Includes first-person terms like *my, me, I* referring to professional background.
   - Mentions or implies something that exists in metadata (even indirectly).
   - Includes first-person terms like my, me, I referring to professional goals, career aspirations, or suitability for roles, even if not in metadata.


   ✅ Examples:
   - “Summarize my projects.”
   - “Can you elaborate on the ones you just mentioned?”
   - “How does my profile match this job?”
   - “Highlight my key strengths.”

2. **WEBSEARCH**
   Choose this if the question:
   - Requires **fresh, external, or time-sensitive knowledge** (e.g., current trends, 2025+ updates, market data, salaries, companies, or technologies not in metadata).
   - Needs information unavailable in the provided resume/job description.

   ✅ Examples:
   - “What are the current AI hiring trends?”
   - “How much do data scientists earn in Canada in 2025?”
   - “Find recent papers on LLM optimization.”

3. **LLM**
   Choose this if the question:
   - Is **general, open-ended, or conversational**.
   - Involves **non-career or personal** queries (e.g., greetings, creative writing, generic advice).
   - Refers to prior conversation but not to resume/job data.
   - Requires **reasoning or discussion**, not factual retrieval.

   ✅ Examples:
   - “Hey, how are you?”
   - “Give me a motivational quote.”
   - “Explain why deep learning is important.”

---

### 🧠 Important:
- **If unsure**, prefer **VECTORSTORE** when the question seems to relate to user’s resume, skills, or projects — even indirectly.
- Use **conversation history** to resolve ambiguous follow-ups (like “these,” “it,” “that one,” etc.).
- Output only **one uppercase word**: VECTORSTORE, WEBSEARCH, or LLM.

---

### 🗣️ User Question
{user_question}

### 🔚 Output:
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

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "job_description", "question", "conversation_history"],
    template="""
You are an **Executive Career Advisor and AI Resume Consultant**. 
Your role is to provide **accurate, contextual, and professional insights** strictly based on the retrieved information.

---

### Context Use Rules
1. **Primary Source** — Use the following retrieved content as your main source of truth:
   {context}

2. **Job Alignment (if applicable)** — If the user’s question relates to a job fit, skills gap, or recommendation:
   - Use this job description: {job_description}
   - Focus on clear, evidence-based comparisons.

3. **Conversation Continuity**
   The following is part of the ongoing conversation. Use it only to maintain context, tone, and continuity — not as factual evidence:
   {conversation_history}
   There is no prior conversation. Answer the question independently.

4. **Response Style**
   - Keep responses structured, concise, and to the point.
   - Avoid speculation or assumptions beyond the provided data.
   - If data is insufficient, clearly say:
     > "The provided information is insufficient to assess this aspect accurately."

---

### User Question
{question}

### Final Answer
"""
)


LLM_FALLBACK_PROMPT = PromptTemplate(
    input_variables=["question", "conversation_history"],
    template="""
You are a **versatile AI assistant** capable of handling questions across any topic — from career and technology to general knowledge, reasoning, and creativity.

---

### Context Continuity
Use the following previous conversation (if any) only to maintain tone, flow, and context — not as a factual source:
{conversation_history}

---

### Instructions
1. **Understand Intent:**
   - If the user asks something career-related, respond professionally with structured insight.
   - If the question is general, curious, or conversational, respond naturally and engagingly.
   - If it’s open-ended or philosophical, give a thoughtful, well-reasoned answer.

2. **Avoid Hallucination:**
   - Answer only with knowledge you are confident about.
   - If the question cannot be answered factually, say so briefly and clearly.

3. **Tone Guidelines:**
   - Professional when discussing work, skills, or learning.
   - Friendly and approachable for general chat or everyday topics.
   - Creative or imaginative when the question invites it.

4. **Response Structure:**
   - 1–3 concise paragraphs.
   - Use examples or reasoning only when it adds clarity.

---

### User Question
{question}

### Answer
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
llm_fallback_runnable = RunnableSequence(LLM_FALLBACK_PROMPT | llm)
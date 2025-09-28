from langgraph.graph import END, StateGraph, START 
from typing_extensions import TypedDict
from typing import List
from rag_chain import build_rag_chain
from routing import build_routing_chain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from config import MODEL_NAME, TEMPERATURE
from langchain_community.chat_models import ChatOpenAI

# Graph State
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    job_description: str 

# Grader Schema
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="yes/no if it answers the question")
    hallucination: str = Field(description="yes/no if contains unsupported info")

parser = PydanticOutputParser(pydantic_object=GradeAnswer)

def expand_query(user_query: str, num_variations: int = 5):
    """
    Generate multiple perspectives for a single query using LLM.
    """
    prompt = f"""
    Rephrase the following query into {num_variations} different ways.
    Each variation should ask the question from a slightly different perspective.
    
    Query: "{user_query}"
    
    Return as a numbered list.
    """
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    response = llm.predict(prompt)
    variations = []
    for line in response.split("\n"):
        line = line.strip()
        if line:
            parts = line.split(". ", 1)
            if len(parts) == 2:
                variations.append(parts[1].strip())
            else:
                variations.append(line)
    return [v for v in variations if v]


def build_workflow(vector_db, web_search_tool):
    workflow = StateGraph(GraphState)
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    rag_chain = build_rag_chain()
    routing_chain = build_routing_chain()

    # ---- Nodes ----  
    # def retrieve(state):
    #     question = state["question"]
    #     docs = vector_db.similarity_search(question)
    #     return {"question": question, "documents": docs}
    def retrieve(state):
        question = state["question"]

        # Expand query into multiple perspectives
        expanded_queries = expand_query(question, num_variations=5)

        # Retrieve docs for each perspective
        retrieved_docs = []
        for q in expanded_queries:
            docs = vector_db.similarity_search(q, k=3)
            retrieved_docs.extend(docs)

        # Remove duplicate documents
        unique_docs = {d.page_content: d for d in retrieved_docs}.values()

        return {"question": question, "documents": list(unique_docs)}

    def generate(state):
        query = state["question"]
    
        # Retrieve top 3 matching docs from vectorstore
        context_docs = vector_db.similarity_search(query, k=3)
        context_text = "\n".join([d.page_content for d in context_docs])
        
        jd = state.get("job_description", "")
        if hasattr(jd, "content"):  # AIMessage -> get the string
            jd = jd.content
        jd = jd.strip() if jd else ""

        # Build LLM input
        if jd:
            llm_input = f"Job Description:\n{jd}\n\nResume Info:\n{context_text}"
        else:
            llm_input = f"Resume Info:\n{context_text}"
        
        # Run LLM chain
        response = rag_chain.run(context=llm_input, question=query)
    
        return {"question": query, "generation": response}

    def llm_fallback(state):
        question = state["question"]
        generation = llm.predict(f"Answer the question/ greeting/ small talk concisely : {question}")
        return {"question": question, "generation": generation}
    
    def web_search(state):
        question = state["question"]
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        prompt = f"""
        Question: {question}
        Context (from web): {web_results}

        Answer concisely.
        """
        generation = llm.predict(prompt)
        return {"question": question, "generation": generation}

    def grade_generation(state):
        question = state["question"]
        generation = state["generation"]
        context = "\n".join([d.page_content for d in state.get("documents", [])])  # optional
        
        prompt_text = f"""
        You are a grader assessing an AI answer.

        1. Check if the answer addresses/resolves the user's question.  
        2. Check if the answer contains hallucinations (information not present in the context/resume provided).  

        Respond ONLY in JSON using this schema:
        {parser.get_format_instructions()}

        User Question: {question}
        AI Answer: {generation}
        Context (resume/docs): {context}
        """

        response = llm.predict(prompt_text)
        graded = parser.parse(response)

        if graded.binary_score.lower() == "yes" and graded.hallucination.lower() == "no":
            return "useful"
        elif graded.binary_score.lower() == "yes" and graded.hallucination.lower() == "yes":
            return "not supported"
        else:
            return "not useful"

    def route_question(state):
        question = state["question"].lower()
        jd_summary = state.get("job_description", "")
        if hasattr(jd_summary, "content"):
            jd_summary = jd_summary.content
        jd_summary = jd_summary.lower() if jd_summary else ""

        if jd_summary and any(word in question for word in ["company", "role", "position", "salary", "location"]):
            return "retrieve"  # VECTORSTORE / generate_from_docs node

        # Otherwise use LLM chain routing
        route = routing_chain.run({
            "user_question": state["question"],
            "metadata_summary": state.get("metadata_summary", ""),
            "jd_summary": state.get("job_description", "")
        }).strip().lower()

        if route == "vectorstore":
            return "retrieve"
        elif route == "websearch":
            return "web_search"
        else:
            return "llm_fallback"

    # ---- Graph ----
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("web_search", web_search)
    workflow.add_node("llm_fallback", llm_fallback)

    workflow.add_conditional_edges(
        START, 
        route_question,
        {
            "retrieve": "retrieve", 
            "web_search": "web_search", 
            "llm_fallback": "llm_fallback"
        }
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges(
        "generate", 
        grade_generation,
        {
            "useful": END, 
            "not useful": "web_search", 
            "not supported": "generate"
        }
    )
    workflow.add_edge("llm_fallback", END)
    workflow.add_edge("web_search", END)

    return workflow.compile()

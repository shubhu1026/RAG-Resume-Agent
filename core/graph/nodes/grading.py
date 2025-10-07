from core.prompts import grade_generation_runnable
from core.graph.types import GradeAnswer
from langchain.output_parsers import PydanticOutputParser

# Parser from BaseModel
parser = PydanticOutputParser(pydantic_object=GradeAnswer)

def grade_generation(state):
    """
    Grade AI answer based on context and question.
    Returns one of: "useful", "not useful", "not supported"
    """
    question = state["question"]
    generation = state["generation"]
    context = "\n".join([d.page_content for d in state.get("documents", [])])  # optional

    # Invoke RunnableSequence
    output = grade_generation_runnable.invoke({
        "question": question,
        "generation": generation,
        "context": context,
        "format_instructions": parser.get_format_instructions()
    })

    # Parse into GradeAnswer
    graded = parser.parse(getattr(output, "content", str(output)))

    # Map graded result to workflow path
    if graded.binary_score.lower() == "yes" and graded.hallucination.lower() == "no":
        return "useful"
    elif graded.binary_score.lower() == "yes" and graded.hallucination.lower() == "yes":
        return "not supported"
    else:
        return "not useful"

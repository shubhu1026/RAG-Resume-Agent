# RAG Resume Assistant

**AI-powered assistant to analyze resumes, compare skills with job descriptions, and provide personalized recommendations using an Adaptive Retrieval-Augmented Generation (RAG) workflow.**

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/shubham1026/rag-resume-assistant)  

---

## Table of Contents
1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Supported File Types](#supported-file-types)  
4. [Workflow](#workflow)  
5. [Adaptive RAG Workflow](#adaptive-rag-workflow)  
6. [Example Use Cases](#example-use-cases)  
7. [Getting Started](#getting-started)  
8. [Environment Variables](#environment-variables)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## Overview
The **RAG Resume Assistant** helps users optimize resumes, compare skills with job descriptions, and interactively query resumes via a chatbot. It is built with **LangChain**, **Gradio UI**, **OpenAI APIs**, and **Tavily** for contextual web retrieval.  

The system ensures **actionable ATS-focused recommendations**, dynamic skill comparisons, and an intelligent chatbot for personalized queries.  

---

## Key Features
- Compare **resume skills** with a job description and generate improvement suggestions.  
- Redact sensitive personal information from resumes automatically.  
- Provide **ATS-focused recommendations** to improve resume visibility.  
- Chatbot interface powered by **Adaptive RAG** for answering questions about resumes, job fit, or general queries.  
- Dynamically retrieves context from resume, web, or LLM depending on the question.  

---

## Supported File Types
- PDF (.pdf)  
- Microsoft Word (.docx)  

---

## Workflow

### 1. Upload
- Upload your **resume** and optionally a **job description**.  

### 2. Resume Processing
- Redact personal information.  
- Clean and prepare document for semantic retrieval:  
  - Section-based  
  - Item-based  
  - Full resume for embeddings.  
- Create a **vector store** for semantic search.  

### 3. Job Description Processing
- Summarize and clean the job description text.  

### 4. Skills Workflow
- Compare resume skills with job description.  
- Generate actionable suggestions for ATS optimization.  

### 5. Chatbot Interface
- Users can ask questions about their resume, projects, experience, or job fit.  
- Uses **Adaptive RAG workflow** to generate context-aware responses.  

---

## Adaptive RAG Workflow
The **Adaptive RAG workflow** intelligently determines the best source for answering user questions:

1. **Question Analysis**: Determine best route – `vectorstore`, `web`, or `LLM`.  
   - Vectorstore: Most resume-related queries (skills, projects, experience, job fit).  
   - Web: Recent trends or external context.  
   - LLM: General questions, greetings, or fallback queries.  

2. **Vectorstore Path**
   - Expand the query for multiple perspectives.  
   - Retrieve documents for both original and expanded queries.  
   - Generate answer using unique retrieved documents.  
   - Grade the generation and decide whether to use it, regenerate, or fallback to web search.
   ```python
   # Map graded result to workflow path
   if graded.binary_score.lower() == "yes" and graded.hallucination.lower() == "no":
       return "useful"
   elif graded.binary_score.lower() == "yes" and graded.hallucination.lower() == "yes":
       return "not supported"
   else:
       return "not useful"
    ```

3. **Web Search Path**
   - Use **Tavily** to fetch contextual information.  
   - Pass context to LLM to generate enhanced answers.  

4. **LLM Path**
   - For general questions, greetings, or explanations.  

---

## Example Use Cases
- **Job seekers** optimizing their resumes for specific roles.  
- **HR professionals** quickly analyzing candidate skills.  
- **Students or professionals** preparing for career transitions.  

---

## Getting Started
### Requirements
- Python 3.10+  
- `pip install -r requirements.txt`  
- OpenAI API key and Tavily API key in `.env`  

### Installation
```bash
git clone https://github.com/yourusername/rag-resume-assistant.git
cd rag-resume-assistant
pip install -r requirements.txt
```

### Running the App
```python
python app.py
```

### Environment Variables
Create a .env file in the project root:
```python
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_PROJECT=RAG-Resume-Agent
```
Optional: Add LangSmith keys for tracing the workflow.

### Contributing
- Fork the repository.
- Create a feature branch.
- Submit a pull request with detailed description.

  

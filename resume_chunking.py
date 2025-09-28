import re
from langchain.schema import Document

def print_docs(docs):
    for i, doc in enumerate(docs, 1):
        print("="*60)
        print(f"📄 Document {i}")
        print(f"🔖 Metadata: {doc.metadata}")
        print("📝 Content:")
        print(doc.page_content.strip())
        print("="*60, "\n")

def chunk_resume(resume_text: str, metadata: dict = None):
    """
    Chunk resume into macro (section) and micro (items) using section headers from metadata.
    """

    print("----------------Metadata--------------\n ", metadata)

    if metadata is None:
        metadata = {}

    docs = []

    # Normalize headers from metadata
    sections_raw = metadata.get("section_headers", "")
    if isinstance(sections_raw, str):
        section_headers = [s.strip() for s in sections_raw.split(",") if s.strip()]
    else:
        section_headers = [s.strip() for s in sections_raw]
    section_headers = [h.upper() for h in section_headers]
    print("----------------Section Headers--------------\n", section_headers)

    # Build regex pattern to match headers (ignore case, optional colon)
    header_pattern = r'^(' + '|'.join([re.escape(h) for h in section_headers]) + r'):?\s*$'
    matches = list(re.finditer(header_pattern, resume_text, flags=re.MULTILINE | re.IGNORECASE))

    if not matches:
        # fallback: return entire resume as one doc
        print("FALLBACK! ENTIRE RESUME IS CHUNKED AS ONE DOC----------------------------------------------------------")
        docs.append(Document(
            page_content=resume_text,
            metadata={**metadata, "section": "full", "type": "section"}
        ))
        return docs

    # Iterate over matches to split sections
    for i, match in enumerate(matches):
        start_idx = match.end()
        end_idx = matches[i+1].start() if i+1 < len(matches) else len(resume_text)

        section_name = match.group(1).lower()
        section_content = resume_text[start_idx:end_idx].strip()

        if section_content:
            docs.extend(_chunk_section(section_content, section_name, metadata))

    return docs

def _chunk_section(section_text: str, section_name: str, metadata: dict):
    """
    Create macro and micro chunks for a single section using structured metadata.
    """
    from langchain.schema import Document

    docs = []

    # Macro chunk: entire section
    docs.append(Document(
        page_content=section_text,
        metadata={**metadata, "section": section_name, "type": "section"}
    ))

    # Micro chunks: structured entries
    items_key = None
    if section_name.lower() in ["projects"]:
        items_key = "projects"
    elif section_name.lower() in ["experience", "professional experience"]:
        items_key = "experience"
    elif section_name.lower() in ["education"]:
        items_key = "education"

    if items_key and metadata.get(items_key):
        # Expect metadata[items_key] to be a list of entries
        entries = metadata[items_key]
        # If it's a string, split by comma
        if isinstance(entries, str):
            entries = [e.strip() for e in entries.split(",") if e.strip()]

        for entry in entries:
            docs.append(Document(
                page_content=entry,
                metadata={**metadata, "section": section_name, "type": "item"}
            ))

    return docs

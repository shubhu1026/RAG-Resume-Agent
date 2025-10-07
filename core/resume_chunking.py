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

    if metadata is None:
        metadata = {}

    docs = []

    docs.append(Document(
        page_content=resume_text.strip(),
        metadata={**metadata, "section": "full", "type": "full_resume"}
    ))

    # Normalize headers from metadata
    sections_raw = metadata.get("section_headers", "")
    if isinstance(sections_raw, str):
        section_headers = [s.strip() for s in sections_raw.split(",") if s.strip()]
    else:
        section_headers = [s.strip() for s in sections_raw]
    section_headers = [h.upper() for h in section_headers]

    # Build regex pattern to match headers (ignore case, optional colon)
    header_pattern = r'^\s*(' + '|'.join([re.escape(h) for h in section_headers]) + r')\s*:?\s*$'
    matches = list(re.finditer(header_pattern, resume_text, flags=re.MULTILINE | re.IGNORECASE))

    if not matches:
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
    from langchain.schema import Document
    import re

    docs = []

    # Macro chunk: entire section
    docs.append(Document(
        page_content=section_text.strip(),
        metadata={"section": section_name, "type": "section"}
    ))

    items_key = None
    if section_name.lower() in ["projects"]:
        items_key = "projects"
    elif section_name.lower() in ["experience", "professional experience"]:
        items_key = "experience"
    elif section_name.lower() in ["education"]:
        items_key = "education"

    if items_key and metadata.get(items_key):
        entries = metadata[items_key]
        if isinstance(entries, str):
            entries = [e.strip() for e in entries.split(",") if e.strip()]

        # Sort entries by length (longest first to avoid partial matches)
        entries = sorted(entries, key=len, reverse=True)

        # Build regex pattern to capture each entry and its following content
        entry_pattern = r'(' + '|'.join(re.escape(e) for e in entries) + r')'
        matches = list(re.finditer(entry_pattern, section_text, flags=re.IGNORECASE))

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(section_text)
            chunk = section_text[start:end].strip()

            docs.append(Document(
                page_content=chunk,
                metadata={"section": section_name, "type": "item"}
            ))

    return docs

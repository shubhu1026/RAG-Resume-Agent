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
    header_pattern = r'^\s*(' + '|'.join([re.escape(h) for h in section_headers]) + r')\s*:?\s*$'
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

# def chunk_resume(resume_text: str, metadata: dict):
    docs = []

    # Extract section headers from metadata
    section_headers_raw = metadata.get("section_headers", "")
    if isinstance(section_headers_raw, str):
        section_headers = [h.strip().upper() for h in section_headers_raw.split(",") if h.strip()]
    else:
        section_headers = [h.strip().upper() for h in section_headers_raw]

    # Build regex to match headers anywhere in text
    header_pattern = r'(?i)^\s*(' + '|'.join([re.escape(h) for h in section_headers]) + r')\s*$'
    matches = list(re.finditer(header_pattern, resume_text, flags=re.MULTILINE))

    # If no matches, fallback to single doc
    if not matches:
        docs.append(Document(
            page_content=resume_text.strip(),
            metadata={**metadata, "section": "full", "type": "section"}
        ))
        return docs

    # Split sections
    sections = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(resume_text)
        header = match.group(1).lower()
        content = resume_text[start:end].strip()
        sections.append((header, content))

    # Build macro and micro chunks
    for header, content in sections:
        # Macro chunk: entire section
        docs.append(Document(
            page_content=content,
            metadata={"section": header, "type": "section"}
        ))

        # Micro chunks: from metadata
        key_map = {
            "projects": "projects",
            "professional experience": "experience",
            "experience": "experience",
            "education": "education"
        }
        key = key_map.get(header)
        if key and metadata.get(key):
            entries = metadata[key]
            # Ensure list
            if isinstance(entries, str):
                entries = [e.strip() for e in entries.split(",") if e.strip()]

            # Use regex to capture full content for each entry
            for entry in entries:
                # Escape for regex
                entry_pattern = re.escape(entry)
                match = re.search(entry_pattern + r'(.+?)(?=\n\s*•|\Z)', content, flags=re.DOTALL | re.IGNORECASE)
                if match:
                    full_text = entry + match.group(1)
                else:
                    full_text = entry
                docs.append(Document(
                    page_content=full_text.strip(),
                    metadata={"section": header, "type": "item"}
                ))

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

        # Prepare regex-safe titles
        safe_titles = [re.escape(e.strip()) for e in entries]

        # Build a combined regex to capture each entry and its content
        combined_pattern = r'(' + '|'.join(safe_titles) + r')'
        # Split section text by titles
        splits = re.split(combined_pattern, section_text, flags=re.IGNORECASE)

        # re.split returns list: [before first title, first title, content1, title2, content2,...]
        i = 1
        while i < len(splits):
            title = splits[i].strip()
            content = splits[i+1].strip() if i+1 < len(splits) else ""
            full_text = f"{title}\n{content}" if content else title

            docs.append(Document(
                page_content=full_text,
                metadata={"section": section_name, "type": "item"}
            ))
            i += 2  # move to next title/content pair

    return docs

import re

SECTION_HEADERS = [
    "abstract",
    "introduction",
    "method",
    "methodology",
    "approach",
    "experiments",
    "results",
    "discussion",
    "conclusion"
]

def extract_sections(text: str) -> dict:
    """
    Naive section splitter using common headings.
    """
    sections = {}
    current_section = "unknown"
    sections[current_section] = []

    for line in text.split("\n"):
        clean_line = line.lower().strip()

        if clean_line in SECTION_HEADERS:
            current_section = clean_line
            sections[current_section] = []
        else:
            sections[current_section].append(line)

    return {k: "\n".join(v) for k, v in sections.items()}

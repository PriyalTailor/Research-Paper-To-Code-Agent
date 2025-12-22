from agent.paper_loader import load_pdf
from agent.section_extractor import extract_sections

if __name__ == "__main__":
    text = load_pdf("examples/saood2021.pdf")
    sections = extract_sections(text)

    for name, content in sections.items():
        print(f"\n===== {name.upper()} =====\n")
        print(content[:1000])

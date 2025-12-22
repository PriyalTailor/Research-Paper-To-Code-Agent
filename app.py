from agent.paper_loader import load_pdf
from agent.section_extractor import extract_sections
from agent.model_reasoner import extract_model_design
import json

if __name__ == "__main__":
    text = load_pdf("examples/saood2021.pdf")

    sections = extract_sections(text)

    model_design = extract_model_design(sections)

    print("\n====== EXTRACTED MODEL DESIGN ======\n")
    print(json.dumps(model_design, indent=2))

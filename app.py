from agent.paper_loader import load_pdf
from agent.section_extractor import extract_sections
from agent.model_reasoner import extract_model_design
from agent.code_generator import generate_pytorch_code, write_code_files

if __name__ == "__main__":
    text = load_pdf("examples/saood2021.pdf")
    sections = extract_sections(text)

    model_design = extract_model_design(sections)

    code = generate_pytorch_code(model_design)
    write_code_files(code)

    print("\n✅ PyTorch code generated in /generated folder")

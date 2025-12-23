import argparse
from agent.paper_loader import load_pdf
from agent.section_extractor import extract_sections
from agent.model_reasoner import extract_model_design, estimate_confidence,save_paper_summary
from agent.code_generator import generate_pytorch_code, write_code_files

def main():
    parser = argparse.ArgumentParser(
        description="Research Paper → PyTorch Code Generator"
    )
    parser.add_argument(
        "--paper",
        type=str,
        required=True,
        help="Path to research paper PDF"
    )
    args = parser.parse_args()

    try:
        # 1. Load PDF
        text = load_pdf(args.paper)

        # 2. Extract sections
        sections = extract_sections(text)
        summary_path = save_paper_summary(sections)
        print(f"\nPaper summary saved to: {summary_path}")

        # 3. LLM reasoning
        model_design = extract_model_design(sections)

        # 4. Confidence estimation
        confidence = estimate_confidence(model_design)

    except Exception as e:
        print("\nERROR during paper processing:")
        print(e)
        return

    # ----- Normal execution continues if no error -----

    print("\nMODEL EXTRACTION REPORT")
    print(f"Confidence score: {confidence:.2f}")

    if model_design.get("missing_details"):
        print("\nMissing / unclear details:")
        for item in model_design["missing_details"]:
            print(f"- {item}")

    try:
        # 5. Code generation
        code = generate_pytorch_code(model_design)
        write_code_files(code)

    except Exception as e:
        print("\nERROR during code generation:")
        print(e)
        return

    print("\nPyTorch code generated in /generated folder")

if __name__ == "__main__":
    main()

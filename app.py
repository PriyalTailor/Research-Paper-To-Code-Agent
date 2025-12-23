import argparse
from agent.paper_loader import load_pdf
from agent.section_extractor import extract_sections
from agent.model_reasoner import extract_model_design, estimate_confidence
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

        # 3. LLM reasoning
        model_design = extract_model_design(sections)

        # 4. Confidence estimation
        confidence = estimate_confidence(model_design)

    except Exception as e:
        print("\n❌ ERROR during paper processing:")
        print(e)
        return

    # ----- Normal execution continues if no error -----

    print("\n📊 MODEL EXTRACTION REPORT")
    print(f"Confidence score: {confidence:.2f}")

    if model_design.get("missing_details"):
        print("\n⚠️ Missing / unclear details:")
        for item in model_design["missing_details"]:
            print(f"- {item}")

    try:
        # 5. Code generation
        code = generate_pytorch_code(model_design)
        write_code_files(code)

    except Exception as e:
        print("\n❌ ERROR during code generation:")
        print(e)
        return

    print("\n✅ PyTorch code generated in /generated folder")

if __name__ == "__main__":
    main()

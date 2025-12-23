from openai import OpenAI
import json
import os
import re

SYSTEM_PROMPT = """

task:
1. Read extracted sections of a research paper.
2. Identify the core model architecture and training procedure.
3. Output a STRICT JSON object describing the model.

Rules:
- Output ONLY valid JSON
- Do NOT include markdown
- Do NOT include explanations
- If information is missing, infer reasonably

JSON schema:
{
  "model_name": "",
  "architecture": {
    "layers": [],
    "inputs": "",
    "outputs": ""
  },
  "training": {
    "loss": "",
    "optimizer": "",
    "learning_rate": ""
  },
  "notes": ""
}
"""

def estimate_confidence(model_design: dict) -> float:
    """
    Simple heuristic confidence score (0–1).
    """
    missing = model_design.get("missing_details", [])
    base = 1.0
    penalty = 0.1 * len(missing)
    return max(0.3, base - penalty)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _extract_json(text: str) -> dict:
    """
    Safely extract JSON object from LLM output.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM response")

    return json.loads(match.group())

def extract_model_design(sections):
    prompt = f"""
    From the following research paper sections, extract:

    1. Problem statement
    2. Input data description
    3. Model architecture
    4. Loss function
    5. Training details
    6. Missing details (list anything unclear or not specified)

    Return ONLY valid JSON with keys:
    problem_statement, input_data, model_architecture,
    loss_function, training_details, missing_details

    Paper content:
    {sections}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a research paper analysis agent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    return json.loads(content)
    
def save_paper_summary(summary: dict, base_name: str, output_dir="generated"):
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary_path


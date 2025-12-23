from openai import OpenAI
import os
import json
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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _extract_code(text: str) -> str:
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def generate_pytorch_code(model_design: dict) -> dict:
    prompt = f"""
Given the following structured model design, generate PyTorch code.

Create:
1. model.py → model architecture
2. dataset.py → dataset skeleton
3. train.py → training loop

Insert TODO comments for missing details.

Model Design:
{json.dumps(model_design, indent=2)}

Return output in this JSON format:
{{
  "model.py": "...",
  "dataset.py": "...",
  "train.py": "..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    raw = response.choices[0].message.content
    return json.loads(raw)

def write_code_files(code_dict: dict, base_name: str, output_dir="generated"):
    os.makedirs(output_dir, exist_ok=True)

    rename_map = {
        "model.py": f"{base_name}_model.py",
        "dataset.py": f"{base_name}_dataset.py",
        "train.py": f"{base_name}_train.py"
    }

    for original_name, content in code_dict.items():
        filename = rename_map.get(original_name, original_name)
        path = os.path.join(output_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


from openai import OpenAI
import os
import json
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a senior machine learning engineer.
Generate clean, minimal, runnable PyTorch code.
Do not add explanations or markdown.
"""

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

def write_code_files(code_dict: dict, output_dir="generated"):
    os.makedirs(output_dir, exist_ok=True)

    for filename, content in code_dict.items():
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)

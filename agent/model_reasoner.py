from openai import OpenAI
import json
import os
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a senior machine learning researcher.
Return ONLY valid JSON.
Do not add explanations, markdown, or extra text.
If something is missing, write "Not specified".
"""

def _extract_json(text: str) -> dict:
    """
    Safely extract JSON object from LLM output.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM response")

    return json.loads(match.group())

def extract_model_design(paper_sections: dict) -> dict:
    prompt = f"""
Extract the following from the research paper:

- problem_statement
- input_data
- model_architecture
- loss_function
- training_details
- missing_details (as a list)

Return ONLY a JSON object with these exact keys.

Paper Sections:
{json.dumps(paper_sections, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        return _extract_json(raw_output)
    except Exception as e:
        print("\n⚠️ RAW LLM OUTPUT (DEBUG):\n")
        print(raw_output)
        raise e

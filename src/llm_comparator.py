import json
import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

ALLOWED_CONDITIONS = {
    "pneumonia", "pleural_effusion", "pneumothorax", "atelectasis",
    "cardiomegaly",
}
ALLOWED_REPORT_STATUSES = {
    "mentioned", "possible", "not_detected_in_report", "not_mentioned",
}
ALLOWED_RELATIONSHIPS = {
    "consistent", "inconsistent", "model_only", "report_only",
    "no_strong_indication",
}


def build_prompt(report_data):
    return f"""
You are a patient-education assistant explaining a chest X-ray report.

Compare the radiologist's written report with the CNN image-model predictions.
Do not diagnose the patient, claim certainty, recommend treatment, or invent
findings. Use cautious, patient-friendly language.

Allowed conditions: {sorted(ALLOWED_CONDITIONS)}
Allowed report statuses: {sorted(ALLOWED_REPORT_STATUSES)}
Allowed relationships: {sorted(ALLOWED_RELATIONSHIPS)}

Return only valid JSON in exactly this format:
{{
  "report_id": "{report_data['report_id']}",
  "comparisons": [
    {{
      "condition": "pneumonia",
      "report_status": "possible",
      "model_probability": 0.72,
      "relationship": "consistent",
      "evidence": "Exact supporting sentence from the report."
    }}
  ],
  "patient_explanation": "Three short, meaningful paragraphs.",
  "disclaimer": "This information is for educational purposes only and is not a medical diagnosis."
}}

Report findings:
{report_data['findings']}

Report impression:
{report_data['impression']}

Rule-based report detections:
{json.dumps(report_data.get('report_conditions', []), indent=2)}

CNN predictions:
{json.dumps(report_data['cnn_predictions'], indent=2)}
"""


def call_llm(prompt):
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError(
            "boto3 is not installed. Run: pip install -r requirements.txt"
        ) from error

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    model_id = os.getenv("BEDROCK_MODEL_ID")

    if not region:
        raise ValueError("AWS_REGION is missing from the .env file.")
    if not model_id:
        raise ValueError("BEDROCK_MODEL_ID is missing from the .env file.")

    client = boto3.client("bedrock-runtime", region_name=region)
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1200, "temperature": 0.2},
    )
    return response["output"]["message"]["content"][0]["text"]


def parse_llm_response(raw_response):
    warnings = []
    cleaned = raw_response.strip().replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("The language model did not return a JSON object.")
    if start > 0 or end < len(cleaned) - 1:
        warnings.append("Extra text appeared outside the JSON response.")

    try:
        return json.loads(cleaned[start:end + 1]), warnings
    except json.JSONDecodeError as error:
        raise ValueError(f"The language model returned invalid JSON: {error}") from error


def validate_llm_response(response):
    if not isinstance(response, dict):
        raise ValueError("The language-model response must be a dictionary.")

    required = {"report_id", "comparisons", "patient_explanation", "disclaimer"}
    missing = required - response.keys()
    if missing:
        raise ValueError(f"The language-model response is missing: {sorted(missing)}")
    if not isinstance(response["comparisons"], list):
        raise ValueError("Comparisons must be a list.")

    for item in response["comparisons"]:
        if item.get("condition") not in ALLOWED_CONDITIONS:
            raise ValueError(f"Invalid condition: {item.get('condition')}")
        if item.get("report_status") not in ALLOWED_REPORT_STATUSES:
            raise ValueError(f"Invalid report status: {item.get('report_status')}")
        if item.get("relationship") not in ALLOWED_RELATIONSHIPS:
            raise ValueError(f"Invalid relationship: {item.get('relationship')}")
        probability = item.get("model_probability")
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError("Model probability must be between 0 and 1.")

    return []

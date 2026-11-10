import json
import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


load_dotenv()


ALLOWED_CONDITIONS = {
    "pneumonia",
    "pleural_effusion",
    "pneumothorax",
    "atelectasis",
    "cardiomegaly",
}

ALLOWED_REPORT_STATUSES = {
    "mentioned",
    "possible",
    "not_detected_in_report",
    "not_mentioned",
}

ALLOWED_RELATIONSHIPS = {
    "consistent",
    "inconsistent",
    "model_only",
    "report_only",
    "no_strong_indication",
}


def build_prompt(report_data):
    """Build the prompt sent to Gemma."""

    return f"""
You are a patient-education assistant explaining a chest X-ray report.

Compare the radiologist's written report with the CNN image-model
predictions. Then write a clear, patient-friendly explanation.

Do not diagnose the patient. Do not say that the patient definitely has a
condition. Do not invent findings that are not supported by the report or
model predictions.

Allowed conditions:
- pneumonia
- pleural_effusion
- pneumothorax
- atelectasis
- cardiomegaly

Allowed report statuses:
- mentioned
- possible
- not_detected_in_report
- not_mentioned

Allowed relationships:
- consistent
- inconsistent
- model_only
- report_only
- no_strong_indication

Interpret uncertainty carefully:
- "may represent" means possible.
- "cannot be excluded" means possible.
- "no evidence of" means not_detected_in_report.
- A condition mentioned in the report is not automatically confirmed.

The patient-facing explanation must contain three meaningful paragraphs:

Paragraph 1:
Explain what the radiology report says in simple language and compare it
with the CNN predictions. Clearly state whether the report and model appear
consistent, inconsistent, or whether one source suggests a finding that the
other does not mention.

Paragraph 2:
Explain the possible condition or conditions in plain language. For example,
explain that pleural effusion means extra fluid around the lung or that
cardiomegaly means the heart appears larger than usual.

Paragraph 3:
Provide general educational context, such as common causes or associations,
only when appropriate. Do not claim to know the cause for this patient.
Explain that symptoms, medical history, physical examination, and additional
testing may be needed for interpretation. Tell the patient to discuss the
results with a qualified healthcare professional.

Use general language such as "can be associated with" rather than saying
that a specific cause is present.

Do not recommend medication or treatment.

Return only valid JSON in exactly this format:

{{
  "report_id": "{report_data["report_id"]}",
  "comparisons": [
    {{
      "condition": "pneumonia",
      "report_status": "possible",
      "model_probability": 0.78,
      "relationship": "consistent",
      "evidence": "Exact supporting sentence from the report."
    }}
  ],
  "patient_explanation": "First paragraph.\\n\\nSecond paragraph.\\n\\nThird paragraph.",
  "disclaimer": "This information is for educational purposes only and is not a medical diagnosis."
}}

Report findings:
{report_data["findings"]}

Report impression:
{report_data["impression"]}

CNN predictions:
{json.dumps(report_data["cnn_predictions"], indent=2)}
"""


def call_llm(prompt):
    """Send the prompt to Gemma through Amazon Bedrock."""

    region = os.getenv("AWS_REGION")
    model_id = os.getenv("BEDROCK_MODEL_ID")

    if not region:
        raise ValueError("AWS_REGION is missing from the .env file.")

    if not model_id:
        raise ValueError(
            "BEDROCK_MODEL_ID is missing from the .env file."
        )

    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=region
    )

    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": 1200,
            "temperature": 0.2
        }
    )

    return response["output"]["message"]["content"][0]["text"]


def parse_llm_response(raw_response):
    """
    Convert Gemma's response into a Python dictionary.

    Returns:
        tuple: parsed response and a list of actual warnings
    """

    warnings = []
    cleaned_response = raw_response.strip()

    # Remove Markdown code fences without treating them as a warning.
    if "```" in cleaned_response:
        cleaned_response = cleaned_response.replace("```json", "")
        cleaned_response = cleaned_response.replace("```", "")
        cleaned_response = cleaned_response.strip()

    start_index = cleaned_response.find("{")
    end_index = cleaned_response.rfind("}")

    if start_index == -1 or end_index == -1:
        raise ValueError("Gemma did not return a valid JSON object.")

    if start_index > 0 or end_index < len(cleaned_response) - 1:
        warnings.append(
            "Extra text appeared outside the JSON response."
        )

    json_text = cleaned_response[start_index:end_index + 1]

    try:
        parsed_response = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Gemma returned invalid JSON: {error}"
        ) from error

    return parsed_response, warnings


def validate_llm_response(response):
    """
    Validate the structure and safety of the LLM response.

    Returns:
        list: warnings found during validation
    """

    warnings = []

    if not isinstance(response, dict):
        raise ValueError("LLM response must be a dictionary.")

    required_fields = {
        "report_id",
        "comparisons",
        "patient_explanation",
        "disclaimer",
    }

    missing_fields = required_fields - response.keys()

    if missing_fields:
        raise ValueError(
            f"LLM response is missing: {sorted(missing_fields)}"
        )

    if not isinstance(response["comparisons"], list):
        raise ValueError("Comparisons must be a list.")

    for comparison in response["comparisons"]:
        condition = comparison.get("condition")
        report_status = comparison.get("report_status")
        relationship = comparison.get("relationship")
        probability = comparison.get("model_probability")

        if condition not in ALLOWED_CONDITIONS:
            raise ValueError(f"Invalid condition: {condition}")

        if report_status not in ALLOWED_REPORT_STATUSES:
            raise ValueError(
                f"Invalid report status: {report_status}"
            )

        if relationship not in ALLOWED_RELATIONSHIPS:
            raise ValueError(
                f"Invalid relationship: {relationship}"
            )

        if not isinstance(probability, (int, float)):
            raise ValueError(
                "Model probability must be a number."
            )

        if not 0 <= probability <= 1:
            raise ValueError(
                "Model probability must be between 0 and 1."
            )

        if not comparison.get("evidence"):
            warnings.append(
                f"No evidence was provided for {condition}."
            )

    explanation = response["patient_explanation"]

    if not isinstance(explanation, str):
        raise ValueError(
            "patient_explanation must be a string."
        )

    paragraph_count = len(
        [paragraph for paragraph in explanation.split("\n\n")
         if paragraph.strip()]
    )

    if paragraph_count < 3:
        warnings.append(
            "The patient explanation contains fewer than three paragraphs."
        )

    disclaimer = response["disclaimer"].lower()

    if "not a medical diagnosis" not in disclaimer:
        raise ValueError(
            "The required medical disclaimer is missing."
        )

    return warnings


if __name__ == "__main__":
    test_input = {
        "report_id": "TEST-001",
        "findings": (
            "Mild opacity is present in the left lower lung. "
            "No pleural effusion or pneumothorax is identified."
        ),
        "impression": (
            "The left lower-lobe opacity may represent pneumonia."
        ),
        "cnn_predictions": {
            "pneumonia": 0.78,
            "pleural_effusion": 0.21,
            "pneumothorax": 0.04,
            "atelectasis": 0.12,
            "cardiomegaly": 0.08,
        },
    }

    try:
        prompt = build_prompt(test_input)
        raw_response = call_llm(prompt)

        parsed_response, parsing_warnings = parse_llm_response(
            raw_response
        )

        validation_warnings = validate_llm_response(
            parsed_response
        )

        all_warnings = parsing_warnings + validation_warnings

        for warning in all_warnings:
            print(f"WARNING: {warning}")

        print("\nPatient Explanation:\n")
        print(parsed_response["patient_explanation"])

        print("\nDisclaimer:\n")
        print(parsed_response["disclaimer"])

    except ClientError as error:
        print(f"ERROR: Bedrock request failed: {error}")

    except Exception as error:
        print(f"ERROR: {error}")
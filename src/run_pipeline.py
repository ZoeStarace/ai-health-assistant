import json

from pdf_extractor import process_pdf
from llm_comparator import (
    build_prompt,
    call_llm,
    parse_llm_response,
    validate_llm_response,
)


FAKE_CNN_PREDICTIONS = {
    "pneumonia": 0.78,
    "pleural_effusion": 0.21,
    "pneumothorax": 0.04,
    "atelectasis": 0.12,
    "cardiomegaly": 0.08,
}


def run_pipeline(pdf_path):
    """Process one PDF and generate a patient-friendly explanation."""

    # Step 1: Extract the report from the PDF
    report = process_pdf(pdf_path)

    # Step 2: Add temporary CNN predictions
    report_data = {
        **report,
        "cnn_predictions": FAKE_CNN_PREDICTIONS,
    }

    # Step 3: Build the prompt for Gemma
    prompt = build_prompt(report_data)

    # Step 4: Call Gemma through Amazon Bedrock
    raw_response = call_llm(prompt)

    # Step 5: Convert Gemma's response into a Python dictionary
    parsed_response, parsing_warnings = parse_llm_response(
        raw_response
    )

    # Step 6: Validate the response
    validation_warnings = validate_llm_response(parsed_response)

    all_warnings = parsing_warnings + validation_warnings

    if all_warnings:
        for warning in all_warnings:
            print(f"WARNING: {warning}")

    # Step 7: Print only the patient-facing explanation
    print("\nPatient Explanation:\n")
    print(parsed_response["patient_explanation"])

    print("\nDisclaimer:\n")
    print(parsed_response["disclaimer"])

    return parsed_response


if __name__ == "__main__":
    pdf_path = "/Users/shreyabharath/Documents/aiHealthAssistant/ai-health-assistant/data/sample_report.pdf"

    try:
        result = run_pipeline(pdf_path)

    except Exception as error:
        print(f"ERROR: {error}")
from pathlib import Path

from .condition_detector import detect_conditions
from .llm_comparator import (
    build_prompt,
    call_llm,
    parse_llm_response,
    validate_llm_response,
)
from .pdf_extractor import process_pdf


DEFAULT_CNN_PREDICTIONS = {
    "pneumonia": 0.72,
    "pleural_effusion": 0.18,
    "pneumothorax": 0.05,
    "atelectasis": 0.36,
    "cardiomegaly": 0.09,
}


PLAIN_LANGUAGE = {
    "pneumonia": "an infection that can cause inflammation in the lungs",
    "pleural_effusion": "extra fluid around a lung",
    "pneumothorax": "air around a lung that may cause it to partly collapse",
    "atelectasis": "an area of the lung that is not fully expanded",
    "cardiomegaly": "a heart that appears larger than usual on the image",
}


def build_local_explanation(detected_conditions, predictions):
    """Create a safe explanation when Bedrock is not configured."""

    detected_by_name = {
        item["condition"]: item for item in detected_conditions
    }
    strongest = max(predictions, key=predictions.get)
    probability = predictions[strongest]

    if detected_conditions:
        report_names = ", ".join(
            item["condition"].replace("_", " ")
            for item in detected_conditions
        )
        first = (
            "The radiology report mentions or discusses "
            f"{report_names}. The temporary image model's highest score is "
            f"for {strongest.replace('_', ' ')} at {probability:.1%}."
        )
    else:
        first = (
            "The report does not clearly mention any of the five conditions "
            "supported by this prototype. The temporary image model's highest "
            f"score is for {strongest.replace('_', ' ')} at {probability:.1%}."
        )

    if strongest in detected_by_name:
        first += " The report and temporary model point to the same general finding."
    else:
        first += " The report and temporary model do not clearly point to the same finding."

    second = (
        f"In general, {strongest.replace('_', ' ')} refers to "
        f"{PLAIN_LANGUAGE[strongest]}. A probability from the model is not a "
        "diagnosis and does not confirm that the condition is present."
    )

    third = (
        "A healthcare professional may consider symptoms, medical history, "
        "physical examination, and other tests when interpreting these results. "
        "Please discuss the original X-ray and radiology report with a qualified "
        "healthcare professional."
    )

    return f"{first}\n\n{second}\n\n{third}"


def run_pipeline(pdf_path, cnn_predictions=None, use_llm=True):
    """Extract a report, compare it with predictions, and return UI-ready data."""

    pdf_path = Path(pdf_path)
    predictions = cnn_predictions or DEFAULT_CNN_PREDICTIONS
    report = process_pdf(pdf_path)

    if not report["findings"] and not report["impression"]:
        raise ValueError("No readable text was found in the uploaded PDF.")

    detected_conditions = detect_conditions(report)
    report_data = {
        **report,
        "report_conditions": detected_conditions,
        "cnn_predictions": predictions,
    }

    warnings = []

    if use_llm:
        try:
            prompt = build_prompt(report_data)
            raw_response = call_llm(prompt)
            parsed_response, parsing_warnings = parse_llm_response(raw_response)
            validation_warnings = validate_llm_response(parsed_response)

            return {
                **parsed_response,
                "report": report,
                "detected_conditions": detected_conditions,
                "warnings": parsing_warnings + validation_warnings,
                "used_llm": True,
            }
        except Exception as error:
            warnings.append(
                "Amazon Bedrock was unavailable, so the local fallback was used: "
                f"{error}"
            )

    return {
        "report_id": report["report_id"],
        "comparisons": [],
        "patient_explanation": build_local_explanation(
            detected_conditions, predictions
        ),
        "disclaimer": (
            "This information is for educational purposes only and is not a "
            "medical diagnosis."
        ),
        "report": report,
        "detected_conditions": detected_conditions,
        "warnings": warnings,
        "used_llm": False,
    }

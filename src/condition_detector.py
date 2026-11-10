import re


CONDITION_PATTERNS = {
    "pneumonia": [
        r"\bpneumonia\b",
    ],
    "pleural_effusion": [
        r"\bpleural effusion\b",
        r"\bpleural fluid\b",
    ],
    "pneumothorax": [
        r"\bpneumothorax\b",
    ],
    "atelectasis": [
        r"\batelectasis\b",
    ],
    "cardiomegaly": [
        r"\bcardiomegaly\b",
        r"\benlarged cardiac silhouette\b",
        r"\benlarged heart\b",
    ],
}


NEGATION_PHRASES = [
    "no",
    "without",
    "negative for",
    "not identified",
    "not seen",
    "absent",
]


UNCERTAINTY_PHRASES = [
    "may represent",
    "may be",
    "could represent",
    "cannot be excluded",
    "possibly",
    "possible",
    "suggestive of",
    "suspicious for",
]


def split_into_sentences(text):
    """Split extracted report text into individual sentences."""
    return re.split(r"(?<=[.!?])\s+", text)


def find_condition_status(sentence, condition):
    """Determine how a condition is described in one sentence."""

    sentence_lower = sentence.lower()

    condition_found = any(
        re.search(pattern, sentence_lower)
        for pattern in CONDITION_PATTERNS[condition]
    )

    if not condition_found:
        return None

    condition_position = min(
        sentence_lower.find(match.group(0))
        for pattern in CONDITION_PATTERNS[condition]
        for match in [re.search(pattern, sentence_lower)]
        if match
    )

    text_before_condition = sentence_lower[:condition_position]

    is_negated = any(
        phrase in text_before_condition
        or phrase in sentence_lower
        and phrase in ["not identified", "not seen"]
        for phrase in NEGATION_PHRASES
    )

    is_uncertain = any(
        phrase in sentence_lower
        for phrase in UNCERTAINTY_PHRASES
    )

    if is_negated:
        status = "not_detected_in_report"
    elif is_uncertain:
        status = "possible"
    else:
        status = "mentioned"

    return {
        "condition": condition,
        "status": status,
        "evidence": sentence.strip(),
    }


def detect_conditions(report):
    """Detect conditions from the report's findings and impression."""

    report_text = (
        f"{report.get('findings', '')} "
        f"{report.get('impression', '')}"
    ).strip()

    sentences = split_into_sentences(report_text)
    detected_conditions = []

    for condition in CONDITION_PATTERNS:
        for sentence in sentences:
            result = find_condition_status(sentence, condition)

            if result is not None:
                detected_conditions.append(result)
                break

    return detected_conditions


if __name__ == "__main__":
    test_report = {
        "findings": (
            "Mild opacity is present in the left lower lung. "
            "No pleural effusion or pneumothorax is identified."
        ),
        "impression": (
            "The left lower-lobe opacity may represent pneumonia."
        ),
    }

    results = detect_conditions(test_report)

    for result in results:
        print(result)
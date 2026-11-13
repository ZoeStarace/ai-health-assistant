import re


CONDITION_PATTERNS = {
    "pneumonia": [r"\bpneumonia\b"],
    "pleural_effusion": [r"\bpleural effusion\b", r"\bpleural fluid\b"],
    "pneumothorax": [r"\bpneumothorax\b"],
    "atelectasis": [r"\batelectasis\b"],
    "cardiomegaly": [
        r"\bcardiomegaly\b",
        r"\benlarged cardiac silhouette\b",
        r"\benlarged heart\b",
    ],
}

NEGATION_PHRASES = [
    "no", "without", "negative for", "not identified", "not seen", "absent"
]
UNCERTAINTY_PHRASES = [
    "may represent", "may be", "could represent", "cannot be excluded",
    "possibly", "possible", "suggestive of", "suspicious for",
]


def split_into_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def find_condition_status(sentence, condition):
    sentence_lower = sentence.lower()
    matches = [
        re.search(pattern, sentence_lower)
        for pattern in CONDITION_PATTERNS[condition]
    ]
    matches = [match for match in matches if match]

    if not matches:
        return None

    condition_position = min(match.start() for match in matches)
    text_before = sentence_lower[:condition_position]

    is_negated = any(phrase in text_before for phrase in NEGATION_PHRASES)
    is_uncertain = any(
        phrase in sentence_lower for phrase in UNCERTAINTY_PHRASES
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
    report_text = (
        f"{report.get('findings', '')} {report.get('impression', '')}"
    ).strip()
    sentences = split_into_sentences(report_text)
    detected = []

    for condition in CONDITION_PATTERNS:
        for sentence in sentences:
            result = find_condition_status(sentence, condition)
            if result is not None:
                detected.append(result)
                break

    return detected

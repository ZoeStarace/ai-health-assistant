import re
from pathlib import Path

import pdfplumber


def extract_pdf_text(pdf_path):
    """Extract text from every page of a PDF."""

    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                pages_text.append(page_text)

    return "\n".join(pages_text).strip()


def extract_report_sections(text):
    """Extract the Findings and Impression sections."""

    findings_match = re.search(
        r"(?:findings|observations?)\s*:?\s*(.*?)(?=\n\s*(?:impression|conclusion)\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    impression_match = re.search(
        r"(?:impression|conclusion)\s*:?\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    findings = findings_match.group(1).strip() if findings_match else ""
    impression = impression_match.group(1).strip() if impression_match else ""

    # If the report has no recognizable headings,
    # preserve all extracted text as the findings.
    if not findings and not impression:
        findings = text.strip()

    return {
        "findings": findings,
        "impression": impression
    }


def process_pdf(pdf_path):
    """Convert a PDF into the standardized internal report format."""

    pdf_path = Path(pdf_path)

    raw_text = extract_pdf_text(pdf_path)
    sections = extract_report_sections(raw_text)

    return {
        "report_id": pdf_path.stem,
        "findings": sections["findings"],
        "impression": sections["impression"]
    }


if __name__ == "__main__":
    report = process_pdf("data/sample_report.pdf")

    print("Extracted report:")
    print(report)
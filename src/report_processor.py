import json
from pathlib import Path


REPORTS_FILE = Path("data/sample_reports.json")


def load_reports():
    """Load all synthetic reports from the JSON file."""
    with open(REPORTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_report(report_id):
    """Find one report using its report ID."""
    reports = load_reports()

    for report in reports:
        if report["report_id"] == report_id:
            return report

    return None


def combine_report_text(report):
    """Combine the findings and impression into one text string."""
    return (
        f"Findings: {report['findings']}\n"
        f"Impression: {report['impression']}"
    )


if __name__ == "__main__":
    report = get_report("TEST-001")

    if report is None:
        print("Report was not found.")
    else:
        print("Report ID:", report["report_id"])
        print(combine_report_text(report))
        print("Conditions listed for testing:", report["report_conditions"])
import io
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.run_pipeline import DEFAULT_CNN_PREDICTIONS, run_pipeline


DISPLAY_NAMES = {
    "pneumonia": "Pneumonia",
    "pleural_effusion": "Pleural Effusion",
    "pneumothorax": "Pneumothorax",
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
}


st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩻",
    layout="centered",
)

st.title("AI Health Assistant")

st.warning(
    "This application is an educational prototype. It does not provide "
    "medical diagnoses or replace professional medical care."
)

st.write(
    "Upload a chest X-ray and its accompanying radiology report to receive "
    "a simplified explanation of the potential findings."
)

st.header("Upload Files")

xray_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg"],
)

report_file = st.file_uploader(
    "Upload a radiology report",
    type=["pdf"],
)

if xray_file is not None:
    try:
        image_bytes = xray_file.getvalue()
        xray_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(xray_image, caption="Uploaded chest X-ray", width=500)
    except Exception as error:
        st.error(f"The uploaded image could not be opened: {error}")

if report_file is not None:
    st.success(f"Report uploaded: {report_file.name}")

if st.button("Analyze Results", type="primary"):
    if xray_file is None:
        st.error("Please upload a chest X-ray first.")
    else:
        # These remain temporary until the trained CNN is connected.
        predictions = DEFAULT_CNN_PREDICTIONS.copy()

        st.header("Potential Image Findings")
        st.caption(
            "Temporary demonstration probabilities are being used until the "
            "trained image model is connected."
        )

        for condition, probability in predictions.items():
            st.write(f"**{DISPLAY_NAMES[condition]}:** {probability:.1%}")
            st.progress(float(probability))

        if report_file is None:
            st.header("Patient-Friendly Explanation")
            strongest = max(predictions, key=predictions.get)
            st.info(
                "The temporary image-model results give the highest score to "
                f"{DISPLAY_NAMES[strongest].lower()}. No report was uploaded, "
                "so the website cannot compare this result with a radiologist's "
                "written findings."
            )
        else:
            temp_path = None
            try:
                # pdfplumber needs a file path, so save the Streamlit upload to
                # a temporary PDF and remove it after processing.
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_pdf:
                    temp_pdf.write(report_file.getvalue())
                    temp_path = Path(temp_pdf.name)

                with st.spinner("Reading and comparing the report..."):
                    result = run_pipeline(
                        temp_path,
                        cnn_predictions=predictions,
                        use_llm=True,
                    )

                st.header("Report Findings")
                report = result["report"]

                with st.expander("View extracted report text"):
                    st.write("**Findings**")
                    st.write(report.get("findings") or "No findings section found.")
                    st.write("**Impression**")
                    st.write(report.get("impression") or "No impression section found.")

                detected = result.get("detected_conditions", [])
                if detected:
                    st.write("**Conditions mentioned in the report:**")
                    for item in detected:
                        name = DISPLAY_NAMES[item["condition"]]
                        status = item["status"].replace("_", " ")
                        st.write(f"- {name}: {status}")
                else:
                    st.caption(
                        "None of the five supported conditions were found in "
                        "the report text."
                    )

                st.header("Patient-Friendly Explanation")
                st.info(result["patient_explanation"])

                if result.get("used_llm"):
                    st.caption("Explanation generated with Amazon Bedrock.")
                else:
                    st.caption(
                        "A safe local explanation was shown because Amazon "
                        "Bedrock was unavailable."
                    )

                warnings = result.get("warnings", [])
                if warnings:
                    with st.expander("Technical warnings"):
                        for warning in warnings:
                            st.write(f"- {warning}")

            except Exception as error:
                st.error(f"The report could not be analyzed: {error}")
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

        st.error(
            "Do not use this result to make medical decisions. Consult a "
            "qualified healthcare professional for medical guidance."
        )

import streamlit as st
from PIL import Image
import io


st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩻",
    layout="centered"
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
    type=["png", "jpg", "jpeg"]
)

report_file = st.file_uploader(
    "Upload a radiology report",
    type=["pdf"]
)

if xray_file is not None:
    try:
        image_bytes = xray_file.getvalue()
        xray_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.image(
            xray_image,
            caption="Uploaded chest X-ray",
            width=500
        )

    except Exception as error:
        st.error(f"The uploaded image could not be opened: {error}")

if report_file is not None:
    st.success(f"Report uploaded: {report_file.name}")

if st.button("Analyze Results", type="primary"):

    if xray_file is None:
        st.error("Please upload a chest X-ray first.")

    else:
        # These are temporary results for testing the website.
        predictions = {
            "Pneumonia": 0.72,
            "Pleural Effusion": 0.18,
            "Pneumothorax": 0.05,
            "Atelectasis": 0.36,
            "Cardiomegaly": 0.09
        }

        st.header("Potential Image Findings")

        for condition, probability in predictions.items():
            st.write(f"**{condition}:** {probability:.1%}")
            st.progress(probability)

        st.header("Patient-Friendly Explanation")

        st.info(
            "The image model identified patterns that may be associated with "
            "pneumonia. This result is uncertain and should be reviewed by a "
            "qualified healthcare professional."
        )

        if report_file is None:
            st.caption(
                "No radiology report was uploaded, so this result is based "
                "only on the chest X-ray."
            )

        st.error(
            "Do not use this result to make medical decisions. Consult a "
            "qualified healthcare professional for medical guidance."
        )
# AI Health Assistant

AI Health Assistant is a multimodal healthcare project that helps users better understand chest X-rays and accompanying radiology reports. The system will analyze a chest X-ray for potential thoracic abnormalities, process the written findings, and present the combined information in patient-friendly language.

## Planned Features

* Upload a chest X-ray in PNG, JPG, or JPEG format
* Upload an accompanying radiology report in PDF format
* Analyze the X-ray for:

  * Pneumonia
  * Pleural effusion
  * Pneumothorax
  * Atelectasis
  * Cardiomegaly
* Extract written findings from the uploaded report
* Compare the image-model predictions with the report findings
* Display prediction probabilities and a simplified explanation
* Clearly communicate uncertainty and medical limitations

## Model Direction

The image component will use a convolutional neural network, such as DenseNet121, to perform multi-label classification. Multi-label classification is necessary because one chest X-ray may contain more than one abnormality.

The report component will extract written findings from a PDF and produce a patient-friendly summary. The image predictions and report findings will then be combined in the Streamlit interface.

## Dataset

The project plans to use the public NIH Chest X-rays dataset available through Kaggle. The dataset contains labeled chest X-ray images for multiple thoracic abnormalities.

The team will focus on five conditions:

1. Pneumonia
2. Pleural effusion
3. Pneumothorax
4. Atelectasis
5. Cardiomegaly

The dataset itself will not be uploaded to this repository because of its size.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/ai-health-assistant.git
cd ai-health-assistant
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application should open at:

```text
http://localhost:8501
```
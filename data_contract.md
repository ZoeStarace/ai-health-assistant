# AI Health Assistant Data Contract

## Conditions

The system will analyze:

- pneumonia
- pleural_effusion
- pneumothorax
- atelectasis
- cardiomegaly

## Image Model Output

The image model will return a probability between 0 and 1 for each condition.
Doesn't have to add up to 1 because we are using multi-label classification so model can detect more than 1 condition. 

```json
{
  "image_id": "TEST-001",
  "predictions": {
    "pneumonia": 0.78,
    "pleural_effusion": 0.21,
    "pneumothorax": 0.04,
    "atelectasis": 0.12,
    "cardiomegaly": 0.08
  }
}
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
model_id = os.getenv("BEDROCK_MODEL_ID")

print("Starting test...")
print("Region:", region)
print("Model:", model_id)

try:
    client = boto3.client(
        "bedrock-runtime",
        region_name=region
    )

    print("Sending request...")

    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": "Say hello in one sentence."}]
            }
        ],
        inferenceConfig={
            "maxTokens": 100,
            "temperature": 0.1
        }
    )

    result = response["output"]["message"]["content"][0]["text"]

    print("Model response:")
    print(result)

except Exception as error:
    print("Request failed:")
    print(type(error).__name__)
    print(error)
    
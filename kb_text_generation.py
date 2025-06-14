import os
import boto3
import time
from dotenv import load_dotenv
from boto3 import client
import json

# Load environment variables from .env file
load_dotenv()

# Read required environment variables
Claude_Model_ID = os.getenv("Claude_Model_ID")
REGION = os.getenv("AWS_REGION")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
DATA_SOURCE_ID = os.getenv("DATA_SOURCE_ID")

if not REGION or not KNOWLEDGE_BASE_ID or not DATA_SOURCE_ID:
    raise ValueError("Missing one or more required environment variables.")


# Create Bedrock clients
runtime_client = boto3.client('bedrock-agent-runtime', region_name=REGION)
mgmt_client = boto3.client('bedrock-agent', region_name=REGION)
model_client = boto3.client("bedrock-runtime", region_name = REGION)

# 1. START INGESTION JOB
print(" Starting ingestion job...")
ingest_response = mgmt_client.start_ingestion_job(
    knowledgeBaseId=KNOWLEDGE_BASE_ID,
    dataSourceId=DATA_SOURCE_ID
)
job_id = ingest_response['ingestionJob']['ingestionJobId']
print(f" Ingestion Job ID: {job_id}")

# 2. WAIT FOR COMPLETION
while True:
    job_status = mgmt_client.get_ingestion_job(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        dataSourceId=DATA_SOURCE_ID,
        ingestionJobId=job_id
    )['ingestionJob']['status']

    print(f" Status: {job_status}")
    if job_status in ['COMPLETE', 'FAILED']:
        break
    time.sleep(5)

if job_status == 'FAILED':
    print(" Ingestion job failed.")
    exit()

print(" Ingestion complete.\n")

def query_kb_and_ask_claude(query_text: str):
    # Step 1: Retrieve context from Knowledge Base
    response = runtime_client.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": query_text},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 1
            }
        }
    )

    print(f" Querying KB: {query_text}")


    # 4. SHOW RESPONSE
    print("\n Retrieved Results:\n")
    for idx, result in enumerate(response['retrievalResults'], start=1):
        content = result['content']['text']
        source = result['location']['s3Location']['uri']
        print(f"{idx}. From: {source}\n---\n{content}\n")

    # Extract retrieved text
    retrieved_text = response["retrievalResults"][0]["content"]["text"]

    # Step 2: Prepare Claude input using Messages API
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "temperature": 0.7,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Use the following context to answer the question in a simple words: Context: {retrieved_text} Question: {query_text}"""
                    }
                ]
            }
        ]
    }

    # Step 3: Call Claude 3 Sonnet via Bedrock
    claude_response = model_client.invoke_model(
        modelId=Claude_Model_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(prompt)
    )

    # Step 4: Parse and return Claude's answer
    response_body = json.loads(claude_response["body"].read())
    return response_body["content"][0]["text"]




if __name__ == "__main__":
    print(f"Claude Model ID: {Claude_Model_ID}")
    print(f"Region: {REGION}")
    print(f"Knowledge Base ID: {KNOWLEDGE_BASE_ID}")
    print(f"Data Sourse ID: {DATA_SOURCE_ID}")
    question = input("Enter any Question?? ")
    answer = query_kb_and_ask_claude(question)
    print("\n Claude's Answer:\n", answer)
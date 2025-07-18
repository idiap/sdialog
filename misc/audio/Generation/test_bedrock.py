import os
from langchain_aws import ChatBedrockConverse

os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "rien"
print(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))

os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "bedrock-api-key-XXX"
print(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))

llm = ChatBedrockConverse(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-east-1",
)

messages = [
    ("human", "Bonjour, qui êtes-vous?"),
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)

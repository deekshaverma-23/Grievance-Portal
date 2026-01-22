import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama

load_dotenv()

llm = Ollama(model="mistral")

print("Local model loaded successfully (mistral via Ollama)")

response = llm.invoke("What is a citizen grievance?")
print("\nDefinition Test:")
print(response)

sentiment_response = llm.invoke(
    "Analyze the sentiment of this text: 'My electricity bill is ridiculously high this month.' "
    "Return only positive, neutral, or negative."
)
print("\nSentiment Test:")
print(sentiment_response)

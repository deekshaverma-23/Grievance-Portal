import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: Google API key not found. Please check your .env file.")
else:
    print("API key loaded successfully.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    response = llm.invoke("What is a citizen grievance?")

    sentiment_response = llm.invoke("Analyze the sentiment of this text: 'My electricity bill is ridiculously high this month.' Is it positive, negative, or neutral?")
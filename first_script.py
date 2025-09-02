import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()

# Access your API key from the environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: Google API key not found. Please check your .env file.")
else:
    print("API key loaded successfully.")

    # Initialize the LLM. Using a model like "gemini-pro"
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

    # Make a simple call to the LLM
    response = llm.invoke("What is a citizen grievance?")

    # Print the response
    print("\nLLM Response:")
    print(response.content)

    # A quick example of sentiment analysis
    sentiment_response = llm.invoke("Analyze the sentiment of this text: 'My electricity bill is ridiculously high this month.' Is it positive, negative, or neutral?")
    print("\nSentiment Analysis:")
    print(sentiment_response.content)
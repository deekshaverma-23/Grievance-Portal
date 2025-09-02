## AI-Powered Citizen Grievance Redressal System 
An intelligent platform designed to automate and optimize the process of handling citizen complaints by leveraging the power of Large Language Models (LLMs) and advanced AI techniques.

## Features
Intelligent Complaint Processing: Automatically analyzes submitted grievances to determine sentiment, categorize them, and assign a priority score.

Knowledge-Augmented Decisions: Utilizes Retrieval-Augmented Generation (RAG) to provide LLMs with relevant government policies and past resolutions, ensuring informed and accurate responses.

Multi-Agent System: A LangGraph-based workflow orchestrates specialized AI agents for sentiment analysis, priority determination, and resolution generation.

Practical Two-Page Interface: A simple user-facing page for complaint submission and a dedicated admin dashboard to view and manage all processed grievances.


Licensed by Google
Persistent Storage: All complaint data, including AI-generated analysis, is stored in a local SQLite database for easy access and logging.

## How It Works
The system operates on a state-of-the-art Generative AI pipeline. A citizen submits a complaint through the Streamlit front-end. This complaint is then passed through a series of chained AI agents:

Sentiment Analysis: An agent determines if the complaint is positive, negative, or neutral.

Priority Agent: Based on the sentiment and content, a priority is assigned (e.g., negative complaints are marked "high priority").

RAG Agent: The complaint is used to search a vector database of government documents and past cases to retrieve relevant context.

Resolution Agent: An LLM, augmented with the retrieved context, generates a comprehensive action plan for the complaint.

Database & Notification: The final analysis and resolution are saved to the database and a notification is simulated to alert officials.

## Setup & Installation
## Prerequisites
Python 3.8+
Google Gemini API Key (available for free via Google AI Studio)

## Steps
1. Clone the Repository -
   git clone https://github.com/<YourUsername>/<YourRepositoryName>.git
   cd <YourRepositoryName>
2. Create and Activate a Virtual Environment -
   python -m venv venv
3. Install Dependencies -
   pip install -r requirements.txt
4. Configure API Key -
   Add your Google API key to the .env file:
   GOOGLE_API_KEY="your_api_key_here"
5. Run the RAG script to create the vector database - 
   python rag_pipeline.py
6. Initialize the Database -
   python database.py
   
## Running the Application
To start the application, simply run the Streamlit command - 
streamlit run app.py

This will launch the app in your browser, where you can navigate between the user complaint submission page and the admin dashboard.

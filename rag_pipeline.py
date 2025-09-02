import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: Google API key not found. Please check your .env file.")
    exit()

# Define the document loader
loader = DirectoryLoader(
    "docs",
    glob="**/*.txt",
    loader_cls=TextLoader
)

docs = loader.load()

# Define the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# Define the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Create the vector store
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

# Create a retriever
retriever = vectorstore.as_retriever()

# Define the RAG prompt template
prompt_template = """
You are an AI assistant for a citizen grievance redressal system. Use the following context to answer the question.
If you don't know the answer, just say that you cannot provide a specific resolution based on the available information.

Question: {question}

Context:
{context}
"""
rag_prompt = ChatPromptTemplate.from_template(prompt_template)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Chain everything together
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Example query from a citizen
citizen_query = "My street has a big pothole near the park. How can this be resolved?"

# Get the response
print("Thinking...")
response = rag_chain.invoke(citizen_query)
print("\nAI Assistant's Response:")
print(response)

# Clean up (optional)
vectorstore.delete_collection()
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: Google API key not found. Please check your .env file.")
    exit()

loader = DirectoryLoader(
    "docs",
    glob="**/*.txt",
    loader_cls=lambda path: TextLoader(path, encoding="utf-8")
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever()

prompt_template = """
You are an AI assistant for a citizen grievance redressal system. Use the following context to answer the question.
If you don't know the answer, just say that you cannot provide a specific resolution based on the available information.

Question: {question}

Context:
{context}
"""
rag_prompt = ChatPromptTemplate.from_template(prompt_template)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

citizen_query = "My street has a big pothole near the park. How can this be resolved?"

print("Thinking...")
response = rag_chain.invoke(citizen_query)
print("\nAI Assistant's Response:")
print(response)

vectorstore.delete_collection()
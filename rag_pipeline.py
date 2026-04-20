import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="mistral")

load_dotenv()

txt_loader = DirectoryLoader(
    "docs",
    glob="**/*.txt",
    loader_cls=TextLoader
)

pdf_loader = DirectoryLoader(
    "docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

docs = txt_loader.load() + pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever()

prompt_template = """
You are an AI assistant for a citizen grievance redressal system.
Use the following context to answer the question concisely.
If you don't know the answer, say you cannot provide a resolution based on available information.

Question: {question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(prompt_template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    citizen_query = "My street has a big pothole near the park. How can this be resolved?"
    print("\n RAG Response:\n")
    response = rag_chain.invoke(citizen_query)
    print(response)
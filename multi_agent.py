import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from database import init_db, save_complaint

class GraphState(TypedDict):
    complaint_text: str
    sentiment: str
    priority: str
    resolution: str
    context_docs: List[str]

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: Google API key not found. Please check your .env file.")
    exit()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever()

def rag_node(state: GraphState):
    complaint_text = state['complaint_text']

    docs = retriever.invoke(complaint_text)
    
    context_docs = [doc.page_content for doc in docs]
    
    return {"context_docs": context_docs}

def sentiment_node(state: GraphState):
    complaint_text = state['complaint_text']
    sentiment_prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following text as positive, neutral, or negative. Do not add any extra words. Just the sentiment.\n\nText: {text}"
    )
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    sentiment_result = sentiment_chain.invoke({"text": complaint_text})
    return {"sentiment": sentiment_result.strip().lower()}

def priority_node(state: GraphState):
    sentiment = state.get('sentiment', 'neutral')
    priority = "high" if sentiment == "negative" else "low"
    return {"priority": priority}

def rag_node(state: GraphState):
    retrieved_docs = ["Sample document about road maintenance policies."]
    return {"context_docs": retrieved_docs}

def resolution_node(state: GraphState):
    complaint_text = state['complaint_text']
    context_docs = state['context_docs']
    
    resolution_prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant. Based on the complaint and context, generate a detailed resolution plan.\n\nComplaint: {complaint_text}\n\nContext: {context_docs}"
    )
    resolution_chain = resolution_prompt | llm | StrOutputParser()
    resolution_result = resolution_chain.invoke({"complaint_text": complaint_text, "context_docs": context_docs})
    return {"resolution": resolution_result}

def save_to_db_node(state: GraphState):
    save_complaint(
        complaint_text=state['complaint_text'],
        sentiment=state['sentiment'],
        priority=state['priority'],
        resolution=state['resolution']
    )
    return {} 

def notify_officials_node(state: GraphState):
    return {} 

workflow = StateGraph(GraphState)

workflow.add_node("sentiment_analysis", sentiment_node)
workflow.add_node("priority_determination", priority_node)
workflow.add_node("rag_retrieval", rag_node)
workflow.add_node("resolution_generation", resolution_node)
workflow.add_node("save_to_db", save_to_db_node)
workflow.add_node("notify_officials", notify_officials_node)

workflow.set_entry_point("sentiment_analysis")
workflow.add_edge("sentiment_analysis", "priority_determination")
workflow.add_edge("priority_determination", "rag_retrieval")
workflow.add_edge("rag_retrieval", "resolution_generation")
workflow.add_edge("resolution_generation", "save_to_db")
workflow.add_edge("save_to_db", "notify_officials")
workflow.add_edge("notify_officials", END)

app = workflow.compile()
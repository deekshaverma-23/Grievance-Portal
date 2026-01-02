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
    severity: str     
    credibility: str   
    priority: str
    category: str
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

def severity_node(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """
        Classify the REAL-WORLD SEVERITY of the complaint based on impact,
        NOT emotional language.

        Severity levels:
        - critical: risk to life, healthcare, fire, accidents
        - high: prolonged disruption of essential services
        - medium: inconvenience affecting daily life
        - low: cosmetic or minor issues

        Complaint: {text}

        Return only one word.
        """
    )
    chain = prompt | llm | StrOutputParser()
    severity = chain.invoke({"text": state['complaint_text']})
    return {"severity": severity.strip().lower()}

def credibility_node(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """
        Analyze the following complaint and determine whether it uses
        exaggeration, sarcasm, or dramatic language disproportionate to the issue.

        Complaint:
        {complaint}

        Return ONLY one of the following words:
        factual
        mildly exaggerated
        highly exaggerated
        """
    )

    chain = prompt | llm | StrOutputParser()

    credibility = chain.invoke({
        "complaint": state["complaint_text"]
    })

    return {"credibility": credibility.strip().lower()}



def impact_signals(text):
    signals = {
        "life_risk": any(w in text for w in ["oxygen", "hospital", "death", "fire"]),
        "duration": any(w in text for w in ["days", "weeks", "hours"]),
        "scale": any(w in text for w in ["entire", "all", "half", "multiple"]),
    }
    return signals


def sentiment_node(state: GraphState):
    complaint_text = state['complaint_text']
    sentiment_prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following text as positive, neutral, or negative. Do not add any extra words. Just the sentiment.\n\nText: {text}"
    )
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    sentiment_result = sentiment_chain.invoke({"text": complaint_text})
    return {"sentiment": sentiment_result.strip().lower()}

def priority_node(state: GraphState):
    severity = state.get("severity", "low")
    credibility = state.get("credibility", "factual")

    if severity == "critical":
        return {"priority": "critical"}
    if severity == "high" and credibility != "highly exaggerated":
        return {"priority": "high"}
    if severity == "medium":
        return {"priority": "medium"}
    return {"priority": "low"}



def resolution_node(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a government grievance resolution system.

        Generate a resolution plan in VALID JSON with EXACTLY these keys:
        - summary (string, one sentence)
        - immediate_actions (array of 2–4 short steps)
        - responsible_department (string)
        - sla_hours (number)

        Complaint: {complaint}
        Context: {context}

        Rules:
        - Do NOT add extra text
        - Do NOT explain the JSON
        - Output ONLY valid JSON
        """
    )

    context = "\n".join(state["context_docs"][:3])

    chain = prompt | llm | StrOutputParser()

    resolution = chain.invoke({
        "complaint": state["complaint_text"],
        "context": context
    })

    return {"resolution": resolution}


def save_to_db_node(state: GraphState):
    save_complaint(
        complaint_text=state['complaint_text'],
        sentiment=state['sentiment'],
        severity=state['severity'],
        credibility=state['credibility'],
        category=state['category'],
        priority=state['priority'],
        resolution=state['resolution']
    )
    return {}


def category_node(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        "Classify the complaint into one category only:\n"
        "Roads, Electricity, Water, Sanitation, Healthcare, Law & Order.\n\n"
        "Complaint: {text}"
    )
    chain = prompt | llm | StrOutputParser()
    category = chain.invoke({"text": state['complaint_text']})
    clean_category = category.replace("**", "").strip()
    return {"category": clean_category}



def notify_officials_node(state: GraphState):
    category = state.get("category", "Unknown")
    priority = state.get("priority", "Unknown")

    print(f"""
    📩 New Complaint Assigned
    Department: {category}
    Priority: {priority}
    """)
    return {}



workflow = StateGraph(GraphState)

workflow.add_node("sentiment_analysis", sentiment_node)
workflow.add_node("severity_assessment", severity_node)
workflow.add_node("credibility_assessment", credibility_node)
workflow.add_node("priority_determination", priority_node)
workflow.add_node("category_classification", category_node)
workflow.add_node("rag_retrieval", rag_node)
workflow.add_node("resolution_generation", resolution_node)
workflow.add_node("save_to_db", save_to_db_node)
workflow.add_node("notify_officials", notify_officials_node)

workflow.set_entry_point("sentiment_analysis")

workflow.add_edge("sentiment_analysis", "severity_assessment")
workflow.add_edge("severity_assessment", "credibility_assessment")
workflow.add_edge("credibility_assessment", "priority_determination")
workflow.add_edge("priority_determination", "category_classification")
workflow.add_edge("category_classification", "rag_retrieval")
workflow.add_edge("rag_retrieval", "resolution_generation")
workflow.add_edge("resolution_generation", "save_to_db")
workflow.add_edge("save_to_db", "notify_officials")
workflow.add_edge("notify_officials", END)


app = workflow.compile()
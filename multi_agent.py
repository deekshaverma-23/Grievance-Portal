import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END

from database import save_complaint

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

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever()

llm = OllamaLLM(model="mistral")

def rag_node(state: GraphState):
    complaint_text = state['complaint_text']
    docs = retriever.invoke(complaint_text)
    context_docs = [doc.page_content for doc in docs]
    return {"context_docs": context_docs}


def severity_node(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """
        Classify severity of the complaint into one of:
        critical, high, medium, low.

        Complaint: {text}

        Return only ONE word.
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": state['complaint_text']})
    return {"severity": result.strip().lower()}

def credibility_node(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a classifier.

        Task: Determine whether the complaint language is exaggerated.

        Allowed outputs (return EXACTLY one):
        factual
        mildly exaggerated
        highly exaggerated

        Rules:
        - Output only ONE of the allowed phrases.
        - Do NOT add punctuation.
        - Do NOT explain.
        - Do NOT add any other text.

        Complaint:
        {complaint}
        """
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"complaint": state["complaint_text"]})

    result = result.strip().lower()

    if "highly" in result:
        return {"credibility": "highly exaggerated"}
    if "mildly" in result:
        return {"credibility": "mildly exaggerated"}
    return {"credibility": "factual"}

def sentiment_node(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a classifier.

        Task: Determine the sentiment of the complaint.

        Allowed outputs (return EXACTLY one word):
        positive
        neutral
        negative

        Rules:
        - Output only ONE of the allowed words.
        - Do NOT add punctuation.
        - Do NOT explain.
        - Do NOT add extra text.

        Complaint:
        {text}
        """
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": state["complaint_text"]})
    result = result.strip().lower()

    if "positive" in result:
        return {"sentiment": "positive"}
    if "neutral" in result:
        return {"sentiment": "neutral"}
    return {"sentiment": "negative"}

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

def category_node(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """
        Select ONE category from:

        Roads, Electricity, Water,
        Sanitation, Healthcare, Law & Order

        Complaint: {text}

        Return ONLY the category name.
        """
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"text": state['complaint_text']}).strip()

    allowed = [
        "Roads", "Electricity", "Water",
        "Sanitation", "Healthcare", "Law & Order"
    ]

    for a in allowed:
        if a.lower() in raw.lower():
            return {"category": a}

    return {"category": "Other"}

def resolution_node(state: GraphState):
    prompt = ChatPromptTemplate.from_template(
        """
        You are an automated government grievance resolution engine.

        You MUST output a VALID JSON object and NOTHING ELSE.

        The JSON must contain EXACTLY these keys:

        {{
        "summary": string,
        "immediate_actions": array of 2 to 4 short strings,
        "responsible_department": string,
        "sla_hours": number
        }}

        Rules:
        - Output ONLY raw JSON.
        - Do NOT include markdown.
        - Do NOT wrap in ``` blocks.
        - Do NOT explain anything.
        - Do NOT add extra keys.
        - Do NOT add trailing text.

        Complaint:
        {complaint}

        Retrieved Context:
        {context}
        """
    )

    context = "\n".join(state["context_docs"][:3])

    chain = prompt | llm | StrOutputParser()

    resolution = chain.invoke(
        {
            "complaint": state["complaint_text"],
            "context": context,
        }
    )

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


def notify_officials_node(state: GraphState):
    # Can be extended later
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

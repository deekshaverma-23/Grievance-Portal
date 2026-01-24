import sys
import os
import json
from pathlib import Path
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from rag_pipeline import retriever 

load_dotenv()

print("--- Initializing Local Models ---")

local_judge = ChatOllama(model="mistral", temperature=0)
judge_llm = LangchainLLMWrapper(local_judge)

hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

gen_llm = OllamaLLM(model="mistral")

def run_rag_pipeline_v2(question: str):
    docs = retriever.invoke(question)
    unique_contexts = []
    seen = set()
    for d in docs:
        content = d.page_content.strip()
        if content not in seen:
            unique_contexts.append(content)
            seen.add(content)
    
    context_text = "\n\n".join(unique_contexts[:5])
    prompt = f"Context: {context_text}\nQuestion: {question}\nOFFICIAL RESPONSE:"
    
    answer = str(gen_llm.invoke(prompt))
    return answer, unique_contexts

DATA_PATH = Path(__file__).parent / "test_grievances.json"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

data_samples = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

print("--- Step 1: Generating Responses ---")
for row in raw:
    ans, ctx = run_rag_pipeline_v2(row["question"])
    data_samples["question"].append(row["question"])
    data_samples["answer"].append(ans)
    data_samples["contexts"].append(ctx)
    data_samples["ground_truth"].append(row["ground_truth"])

dataset = Dataset.from_dict(data_samples)

print("\n--- Step 2: Running Local RAGAS Evaluation ---")

metrics = [
    Faithfulness(llm=judge_llm),
    ContextPrecision(llm=judge_llm),
    ContextRecall(llm=judge_llm),
    AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings)
]
local_config = RunConfig(max_workers=2, timeout=180)

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    run_config=local_config
)

df = result.to_pandas()
out_path = Path(__file__).parent / "RAG_Final_Optimized_Report.csv"
df.to_csv(out_path, index=False)

print("\n" + "="*40)
print("       LOCAL EVALUATION COMPLETE")
print("="*40)
print(df.mean(numeric_only=True))
print(f"\nSaved to: {out_path}")
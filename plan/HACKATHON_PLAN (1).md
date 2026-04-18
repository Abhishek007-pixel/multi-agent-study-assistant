# 🎓 AI Study Assistant — Hackathon Build Plan
> **Time Budget: 6 hours | Team: 3 people | Framework: LangGraph**

---

## 📌 Project Summary

A **Multi-Agent AI Study Assistant** powered by LangGraph where specialized AI agents collaborate to help students learn any topic. The user types a query like _"Teach me photosynthesis and test me"_ and the system automatically explains the concept AND generates a quiz.

**What makes this stand out from basic chatbots:**
- LangGraph state machine with conditional routing (not simple if/elif)
- Two-Stage Retrieval: BM25 + FAISS → RRF fusion → Cross-Encoder reranking
- Separate fast/reasoning paths depending on query type
- Different LLM sizes per agent (cost/speed optimization)

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         Planner Agent           │
│   (LangGraph Conditional Edge)  │
└────────────────┬────────────────┘
                 │
     ┌───────────┴───────────┐
     │ (quick_question)      │ (learn / quiz / learn_and_test)
     ▼                       ▼
┌──────────────┐    ┌─────────────────────────────────────┐
│ Fast Response│    │          Research Agent              │
│ Agent        │    │  Stage 1a: BM25 keyword search       │
│ llama3-8b    │    │  Stage 1b: FAISS vector search       │
└──────┬───────┘    │  RRF Fusion (top 10 combined)        │
       │            │  Stage 2: Cross-Encoder rerank → 3   │
       │            └──────────────────┬──────────────────┘
       │                               │
       │              ┌────────────────┴────────────────┐
       │              ▼                                  ▼
       │    ┌──────────────────┐             ┌──────────────────┐
       │    │ Explanation Agent│             │   Quiz Agent     │
       │    │  llama3-70b      │             │   llama3-70b     │
       │    └────────┬─────────┘             └────────┬─────────┘
       │             │                                 │
       │             └──────────────┬──────────────────┘
       │                            ▼
       │                 ┌──────────────────┐
       │                 │ Synthesizer Node │
       │                 └────────┬─────────┘
       │                          │
       └──────────────────────────┘
                                  │
                                  ▼
                     Final Output (Streamlit UI)
```

---

## 👥 Agent Roles

| Agent | Role | Model | What it does |
|-------|------|-------|--------------|
| **Planner Agent** | Coordinator | llama3-70b | Classifies intent, sets LangGraph routing |
| **Research Agent** | RAG + Rerank | — | BM25 + FAISS → RRF → Cross-Encoder |
| **Explanation Agent** | Tutor | llama3-70b | Simplifies and structures the concept |
| **Quiz Agent** | Examiner | llama3-70b | Generates MCQs and short answer questions |
| **Fast Response Agent** | Quick help | llama3-8b | Direct LLM call, no RAG, low latency |
| **Synthesizer Node** | Combiner | — | Merges explanation + quiz into final output |

---

## 🔍 Two-Stage Retrieval Pipeline (The Core Innovation)

```
User Query
    │
    ├──► BM25 Search (keyword match) ──► Top 10 docs
    │
    ├──► FAISS Search (semantic match) ──► Top 10 docs
    │
    ▼
RRF Fusion (Reciprocal Rank Fusion)
    │  Combines both ranked lists into one unified top 10
    │  Formula: RRF(d) = Σ 1/(k + rank_i(d))  where k=60
    │
    ▼
Cross-Encoder Reranker
    │  Scores all 10 (query, doc) pairs precisely
    │  Returns only top 3 highest-quality chunks
    │
    ▼
Context passed into LangGraph state → LLM agents
```

**Why this matters for judges:**
- BM25 catches exact keyword matches (good for named topics like "mitosis")
- FAISS catches semantic matches (good for paraphrased queries)
- RRF combines both without needing to tune weights
- Cross-Encoder removes irrelevant chunks before they reach the LLM = fewer hallucinations

---

## 🔁 Full Workflow (Step by Step)

1. User types query in Streamlit chat UI
2. **Planner Agent** classifies intent into one of:
   - `learn_only` → research + explain
   - `quiz_only` → research + quiz
   - `learn_and_test` → research + explain + quiz
   - `quick_question` → fast response only, skip RAG entirely
3. LangGraph routes via conditional edge based on intent
4. **Research Agent** runs Two-Stage Retrieval (BM25 + FAISS → RRF → Cross-Encoder)
5. Context is stored in LangGraph state and shared with downstream agents
6. **Explanation Agent** and/or **Quiz Agent** consume the context
7. **Synthesizer Node** combines their outputs into one final response
8. Final response displayed in Streamlit chat UI

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Orchestration | **LangGraph** | State machine, conditional routing, clean agent graph |
| LLM (reasoning) | **Groq llama3-70b-8192** | Free, fast, high quality for explain/quiz |
| LLM (fast path) | **Groq llama3-8b-8192** | Smaller and faster for quick follow-ups |
| Vector Search | **FAISS** | In-memory, zero setup needed |
| Keyword Search | **BM25 (rank_bm25)** | Exact term matching, complements FAISS |
| Rank Fusion | **RRF** | Combines FAISS + BM25 rankings without weight tuning |
| Reranker | **Cross-Encoder ms-marco-MiniLM-L-4-v2** | Filters top 10 → top 3 before LLM |
| Embeddings | **sentence-transformers all-MiniLM-L6-v2** | Free, runs fully locally |
| Frontend | **Streamlit** | Fastest UI for hackathon |
| Language | **Python 3.10+** | Standard |
| Deployment | **Streamlit Cloud** or localhost | Free, 5-minute deploy |

---

## 📦 Dependencies

### `requirements.txt`
```
langchain==0.2.0
langchain-groq==0.1.3
langchain-community==0.2.0
langgraph==0.0.65
faiss-cpu==1.8.0
sentence-transformers==2.7.0
rank-bm25==0.2.2
streamlit==1.35.0
python-dotenv==1.0.1
pypdf==4.2.0
```

### Install command
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
ai-study-assistant/
│
├── app.py                        # Streamlit UI + LangGraph entry point
├── requirements.txt
├── .env                          # API keys — never commit this
│
├── agents/
│   ├── planner_agent.py          # Intent classification
│   ├── explanation_agent.py      # Simplifies and explains concepts
│   ├── quiz_agent.py             # Generates MCQs
│   └── fast_response_agent.py    # Quick direct LLM answer, no RAG
│
├── memory/
│   ├── vector_store.py           # FAISS index builder + search
│   ├── bm25_store.py             # BM25 index builder + search
│   ├── retriever.py              # RRF fusion + Cross-Encoder reranking
│   └── sample_notes/             # Drop .txt or .pdf study notes here
│       ├── biology.txt
│       ├── history.txt
│       └── physics.txt
│
└── graph/
    └── study_graph.py            # LangGraph state, nodes, edges, routing
```

---

## 💻 Code — All Core Files

---

### `.env`
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free key at: https://console.groq.com

---

### `memory/vector_store.py`
```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import os

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents(docs_path="memory/sample_notes"):
    documents = []
    for filename in os.listdir(docs_path):
        path = os.path.join(docs_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        documents.extend(loader.load())
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)

def faiss_search(query: str, vector_store, k=10):
    return vector_store.similarity_search(query, k=k)
```

---

### `memory/bm25_store.py`
```python
from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized = [chunk.page_content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k=10):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in top_indices]
```

---

### `memory/retriever.py`
```python
from sentence_transformers import CrossEncoder
from memory.vector_store import faiss_search

# Use the lightest cross-encoder — fast enough on CPU for hackathon
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2")

def rrf_fusion(faiss_docs, bm25_results, k=60):
    """
    Reciprocal Rank Fusion.
    RRF score = sum of 1/(k + rank) across all lists the doc appears in.
    k=60 is the standard default value.
    """
    scores = {}

    for rank, doc in enumerate(faiss_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    for rank, (doc, _) in enumerate(bm25_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    # Build unified doc map
    all_docs = {doc.page_content: doc for doc in faiss_docs}
    for doc, _ in bm25_results:
        all_docs[doc.page_content] = doc

    # Sort by RRF score descending
    ranked = sorted(all_docs.values(), key=lambda d: scores.get(d.page_content, 0), reverse=True)
    return ranked[:10]

def cross_encoder_rerank(query: str, docs, top_k=3):
    """
    Cross-Encoder scores each (query, doc) pair precisely.
    Returns top_k chunks as a single string for the LLM.
    """
    if not docs:
        return "No relevant content found. Use general knowledge."

    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    best = [doc.page_content for doc, _ in scored[:top_k]]
    return "\n\n".join(best)

def retrieve(query: str, vector_store, bm25_store):
    """
    Full pipeline:
    1. BM25 keyword search → top 10
    2. FAISS semantic search → top 10
    3. RRF fusion → unified top 10
    4. Cross-Encoder rerank → top 3 chunks
    """
    faiss_docs = faiss_search(query, vector_store, k=10)
    bm25_results = bm25_store.search(query, k=10)
    fused = rrf_fusion(faiss_docs, bm25_results)
    return cross_encoder_rerank(query, fused, top_k=3)
```

---

### `agents/planner_agent.py`
```python
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

PLANNER_PROMPT = ChatPromptTemplate.from_template("""
You are a study assistant coordinator. Classify the user's intent.

User query: {query}

Reply with ONLY one of these exact labels (no explanation, no punctuation):
- learn_only
- quiz_only
- learn_and_test
- quick_question

Intent:
""")

def classify_intent(query: str) -> str:
    chain = PLANNER_PROMPT | llm
    result = chain.invoke({"query": query})
    intent = result.content.strip().lower()
    valid = ["learn_only", "quiz_only", "learn_and_test", "quick_question"]
    # Always fallback safely — never crash on unexpected output
    return intent if intent in valid else "learn_only"
```

---

### `agents/explanation_agent.py`
```python
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os

llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

PROMPT = ChatPromptTemplate.from_template("""
You are a friendly tutor. Using the context below, explain the topic clearly.
Write for a high school student.
Structure your answer as: 1) Definition 2) How it works 3) Real-world example.

Topic: {topic}
Context: {context}

Explanation:
""")

def explain(topic: str, context: str) -> str:
    return (PROMPT | llm).invoke({"topic": topic, "context": context}).content
```

---

### `agents/quiz_agent.py`
```python
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os

llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

PROMPT = ChatPromptTemplate.from_template("""
You are a quiz generator. Generate exactly 3 multiple choice questions based on the context.
Each question must have options A, B, C, D and clearly mark the correct answer at the end.

Topic: {topic}
Context: {context}

Quiz:
""")

def generate_quiz(topic: str, context: str) -> str:
    return (PROMPT | llm).invoke({"topic": topic, "context": context}).content
```

---

### `agents/fast_response_agent.py`
```python
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os

# Intentionally smaller and faster model for quick questions
llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

PROMPT = ChatPromptTemplate.from_template("""
Answer the following question in 2-3 sentences. Be direct and concise.

Question: {query}

Answer:
""")

def quick_answer(query: str) -> str:
    return (PROMPT | llm).invoke({"query": query}).content
```

---

### `graph/study_graph.py`
```python
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from agents.planner_agent import classify_intent
from agents.explanation_agent import explain
from agents.quiz_agent import generate_quiz
from agents.fast_response_agent import quick_answer

# ── 1. Define LangGraph State ──────────────────────────────────
class AgentState(TypedDict):
    query: str
    intent: str
    context: str
    explanation: str
    quiz: str
    final_output: str
    vector_store: Any   # FAISS store passed at runtime
    bm25_store: Any     # BM25 store passed at runtime

# ── 2. Node Functions ──────────────────────────────────────────
def planner_node(state: AgentState):
    intent = classify_intent(state["query"])
    return {"intent": intent}

def research_node(state: AgentState):
    from memory.retriever import retrieve
    context = retrieve(state["query"], state["vector_store"], state["bm25_store"])
    return {"context": context}

def explanation_node(state: AgentState):
    exp = explain(state["query"], state["context"])
    return {"explanation": "## 📖 Explanation\n" + exp}

def quiz_node(state: AgentState):
    qz = generate_quiz(state["query"], state["context"])
    return {"quiz": "## 📝 Quiz\n" + qz}

def fast_response_node(state: AgentState):
    ans = quick_answer(state["query"])
    return {"final_output": ans}

def synthesizer_node(state: AgentState):
    parts = []
    if state.get("explanation"):
        parts.append(state["explanation"])
    if state.get("quiz"):
        parts.append("---\n" + state["quiz"])
    return {"final_output": "\n\n".join(parts)}

# ── 3. Routing Logic ───────────────────────────────────────────
def route_after_planner(state: AgentState):
    if state["intent"] == "quick_question":
        return "fast_path"
    return "research_path"

def route_after_research(state: AgentState):
    intent = state["intent"]
    if intent == "learn_only":
        return "explain_only"
    elif intent == "quiz_only":
        return "quiz_only"
    else:
        return "both"

# ── 4. Build the Graph ─────────────────────────────────────────
def build_graph():
    workflow = StateGraph(AgentState)

    # Register all nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("research", research_node)
    workflow.add_node("explain", explanation_node)
    workflow.add_node("quiz", quiz_node)
    workflow.add_node("fast_response", fast_response_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Entry point
    workflow.set_entry_point("planner")

    # Planner → fast path OR research path
    workflow.add_conditional_edges("planner", route_after_planner, {
        "fast_path": "fast_response",
        "research_path": "research"
    })

    # Research → explain only / quiz only / both
    workflow.add_conditional_edges("research", route_after_research, {
        "explain_only": "explain",
        "quiz_only": "quiz",
        "both": "explain"
    })

    # Sequential flow for learn_and_test: explain → quiz → synthesizer
    workflow.add_edge("explain", "quiz")
    workflow.add_edge("quiz", "synthesizer")
    workflow.add_edge("synthesizer", END)
    workflow.add_edge("fast_response", END)

    return workflow.compile()

study_graph = build_graph()
```

---

### `app.py`
```python
import streamlit as st
from dotenv import load_dotenv
from memory.vector_store import load_documents, split_documents, build_vector_store
from memory.bm25_store import BM25Store
from graph.study_graph import study_graph

load_dotenv()

st.set_page_config(page_title="AI Study Assistant", page_icon="🎓", layout="centered")
st.title("🎓 AI Study Assistant")
st.caption("Multi-Agent AI — LangGraph + BM25 + FAISS + Cross-Encoder")

# Load and cache both stores once at startup
@st.cache_resource
def load_stores():
    docs = load_documents()
    chunks = split_documents(docs)
    vs = build_vector_store(chunks)
    bm25 = BM25Store(chunks)
    return vs, bm25

vector_store, bm25_store = load_stores()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Main input
if query := st.chat_input("E.g. 'Teach me photosynthesis and test me'"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Agents working..."):
            result = study_graph.invoke({
                "query": query,
                "intent": "",
                "context": "",
                "explanation": "",
                "quiz": "",
                "final_output": "",
                "vector_store": vector_store,
                "bm25_store": bm25_store
            })
            final = result["final_output"]
            st.markdown(final)

    st.session_state.messages.append({"role": "assistant", "content": final})
```

---

## 🗓️ 6-Hour Build Timeline

| Time | Person 1 | Person 2 | Person 3 |
|------|----------|----------|----------|
| **Hour 1** | Setup: venv, `.env`, folder structure, install deps | Write `vector_store.py` + `bm25_store.py`, test both search functions | Write `planner_agent.py`, test all 4 intents |
| **Hour 2** | Write `retriever.py` (RRF + Cross-Encoder), test full pipeline | Write `explanation_agent.py`, test with 3 topics | Write `quiz_agent.py`, check MCQ quality |
| **Hour 3** | Write `fast_response_agent.py` | Build `graph/study_graph.py` — nodes + edges + routing | Build `app.py` Streamlit UI, connect graph |
| **Hour 4** | End-to-end test all 4 intents | Fix bugs — empty FAISS, bad intent, missing context | UI polish — spinner, chat history, captions |
| **Hour 5** | Add 3 sample notes (biology, history, physics) | Test all demo queries, fix edge cases | Deploy to Streamlit Cloud or confirm localhost |
| **Hour 6** | **BUFFER / DEBUG** | Prepare 3 live demo queries | Prepare 2-minute pitch |

---

## ⚡ Quick Start Commands

```bash
# 1. Setup
mkdir ai-study-assistant && cd ai-study-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Sample notes
mkdir -p memory/sample_notes

echo "Photosynthesis is the process by which plants convert sunlight, water and CO2 into glucose and oxygen. It occurs in chloroplasts. Light-dependent reactions happen in the thylakoid membrane. The Calvin cycle happens in the stroma." > memory/sample_notes/biology.txt

echo "The French Revolution began in 1789. Key causes: financial crisis, social inequality, weak leadership of Louis XVI. Key events: storming of Bastille, Declaration of Rights of Man, Reign of Terror, execution of Louis XVI." > memory/sample_notes/history.txt

echo "Newton's three laws: 1) Object at rest stays at rest. 2) F=ma. 3) Every action has equal and opposite reaction. Gravitational formula: F = G*m1*m2/r^2. Kinetic energy: KE = 0.5*m*v^2." > memory/sample_notes/physics.txt

# 5. Run
streamlit run app.py
```

---

## 🎯 Demo Queries for Judges

| Query | Intent | What judges see |
|-------|--------|-----------------|
| `"What is ATP?"` | `quick_question` | Fast 2-sentence answer, llama3-8b, no RAG |
| `"Teach me the French Revolution"` | `learn_only` | Full RAG → RRF → Cross-Encoder → structured explanation |
| `"Teach me Newton's laws and test me"` | `learn_and_test` | Full pipeline → explanation + 3 MCQs |

---

## 🏆 What to Tell Judges (Your Pitch Points)

- **Two retrieval methods**: BM25 for exact keywords + FAISS for semantic meaning, fused with Reciprocal Rank Fusion
- **Cross-Encoder reranker**: Filters 10 candidates to the 3 most relevant before hitting the LLM — reduces hallucinations
- **LangGraph state machine**: Proper agent orchestration, not just a chatbot with if/else logic
- **Two LLM sizes**: llama3-8b for speed, llama3-70b for quality — optimized per task

---

## 🚨 Pitfalls to Avoid

- **FAISS crashes if no docs** — always have at least 1 sample note file before running
- **Cross-Encoder is slow on large models** — stick to `ms-marco-MiniLM-L-4-v2`, don't switch to larger ones
- **LangGraph KeyError** — always pass ALL state keys in `invoke()` even as empty strings
- **Groq rate limit during demo** — if it hits limits, temporarily switch planner to llama3-8b
- **Intent fallback** — code already defaults to `learn_only` if planner gives unexpected output

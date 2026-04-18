"""Streamlit UI + LangGraph entry point."""

import os

import streamlit as st
from dotenv import load_dotenv

from graph.study_graph import study_graph

load_dotenv()

st.set_page_config(page_title="AI Study Assistant", page_icon="🎓", layout="centered")
st.title("🎓 AI Study Assistant")
st.caption("Multi-Agent AI — LangGraph + BM25 + FAISS + Cross-Encoder")

if not os.getenv("GROQ_API_KEY"):
    st.warning("Copy `.env.example` to `.env` and set `GROQ_API_KEY` before using LLM/RAG paths.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("E.g. 'Teach me photosynthesis and test me'"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Agents working..."):
            result = study_graph.invoke(
                {
                    "query": query,
                    "intent": "",
                    "context": "",
                    "explanation": "",
                    "quiz": "",
                    "final_output": "",
                    "vector_store": None,
                    "bm25_store": None,
                }
            )
            final = result["final_output"]
            st.markdown(final)

    st.session_state.messages.append({"role": "assistant", "content": final})

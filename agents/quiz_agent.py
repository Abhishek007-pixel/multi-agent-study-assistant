"""Quiz agent: MCQs from retrieved context."""

from langchain_core.prompts import ChatPromptTemplate

from agents.groq_llms import chat_groq_70b

QUIZ_PROMPT = ChatPromptTemplate.from_template("""
You are an examiner. Based on the context below, generate exactly 3 multiple-choice questions.
If the context says "Use general knowledge", rely on your own expertise.

Each question must:
- Have 4 options labeled A, B, C, D
- Clearly mark the correct answer
- Include a one-sentence explanation of why it is correct

Topic: {topic}
Context: {context}

Quiz:
""")


def generate_quiz(topic: str, context: str) -> str:
    chain = QUIZ_PROMPT | chat_groq_70b()
    return chain.invoke({"topic": topic, "context": context}).content

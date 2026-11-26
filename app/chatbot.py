from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.config import CHAT_MODEL

prompt = ChatPromptTemplate.from_template(
    """You are a helpful Travel Desk assistant.
Answer the question using ONLY the policy context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}"""
)


def build_chain():
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return prompt | llm | StrOutputParser()


def format_context(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def ask_travel_desk(question, retriever, chain):
    retrieved_docs = retriever.invoke(question)
    context = format_context(retrieved_docs)

    answer = chain.invoke({"context": context, "question": question})

    print("\nTRAVEL DESK CHATBOT")
    print("-" * 55)
    print(answer)

    print("\nSOURCES:")
    for source in sorted({doc.metadata.get("source", "Unknown") for doc in retrieved_docs}):
        print("-", source)

    return answer
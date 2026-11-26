from app.config import check_api_key
from app.vectorstore import get_retriever
from app.chatbot import build_chain, ask_travel_desk


def main():
    check_api_key()

    print("Setting up Travel Desk...")
    retriever = get_retriever()
    chain = build_chain()
    print("Ready! Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue
        ask_travel_desk(question, retriever, chain)


if __name__ == "__main__":
    main()
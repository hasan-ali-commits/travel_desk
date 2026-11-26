from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import EMBEDDING_MODEL, INDEX_DIR, TOP_K
from app.loader import load_and_split


def get_retriever():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if INDEX_DIR.exists():
        # Already built before → just load it from the folder (no API cost)
        vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Loaded saved FAISS index.")
    else:
        # First time → embed all chunks and save them
        chunks = load_and_split()
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        vectorstore.save_local(str(INDEX_DIR))
        print("Built and saved new FAISS index.")

    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
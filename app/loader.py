from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import POLICY_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split():
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Policy file not found: {POLICY_PATH}")

    documents = TextLoader(str(POLICY_PATH), encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    return chunks


# Lets you test this file alone: python -m app.loader
if __name__ == "__main__":
    chunks = load_and_split()
    print(f"Created {len(chunks)} chunks")
    print("First chunk:\n", chunks[0].page_content[:300])
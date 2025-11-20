import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import DOCS_DIR, CHROMA_DB_DIR


def get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    else:
        # Fallback loader that tries to understand many formats
        return UnstructuredFileLoader(file_path)


def load_documents(docs_dir: str):
    all_docs = []
    for root, _, files in os.walk(docs_dir):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                loader = get_loader(file_path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = file_path
                all_docs.extend(docs)
                print(f"Loaded {file_path}")
            except Exception as e:
                print(f"[WARN] Error loading {file_path}: {e}")
    return all_docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    print("Creating embeddings (sentence-transformers/all-mpnet-base-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    print(f"Building Chroma DB at: {CHROMA_DB_DIR}")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    vs.persist()
    print("✅ Vector store built and persisted.")


def run_ingestion():
    print(f"📂 Loading documents from: {DOCS_DIR}")
    docs = load_documents(DOCS_DIR)
    print(f"Loaded {len(docs)} raw documents")

    print("✂️ Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    build_vectorstore(chunks)


if __name__ == "__main__":
    run_ingestion()

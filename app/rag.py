from typing import List, Dict

from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import GROQ_API_KEY, CHAT_MODEL, CHROMA_DB_DIR

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions using ONLY the provided context.
- If the answer is not in the context, reply: "I do not have enough information from the documents to answer that."
- Always mention which source file(s) you used.
- Be concise and clear.
"""

_client = Groq(api_key=GROQ_API_KEY)
_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def retrieve_context(question: str, k: int = 5):
    vs = get_vectorstore()
    docs = vs.similarity_search(question, k=k)
    return docs


def format_docs(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[{i}] Source: {src}\n{d.page_content}")
    return "\n\n".join(blocks)


def answer_question(question: str) -> Dict:
    docs = retrieve_context(question, k=5)
    context_str = format_docs(docs)

    user_prompt = f"""
You MUST use the following context to answer the question.

Context:
{context_str}

Question: {question}

Answer clearly. If the context does not contain the answer, say so.
"""

    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    answer = resp.choices[0].message["content"]

    sources = []
    for d in docs:
        sources.append(
            {
                "source": d.metadata.get("source", "unknown"),
                "preview": d.page_content[:200],
            }
        )

    return {"answer": answer, "sources": sources}

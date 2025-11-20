from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .config import GROQ_API_KEY, CHAT_MODEL, CHROMA_DB_DIR
from .bm25_store import BM25Store

client = Groq(api_key=GROQ_API_KEY)
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

SYSTEM_PROMPT = """
You are a helpful assistant. Follow these rules:

1. If context is provided AND relevant → use it.
2. If context is NOT needed → answer normally.
3. Keep answers short and clear (5-8 sentences max).
4. Do NOT hallucinate. If you don't know, say so.
5. Always return unique source filenames when using RAG.
"""


# --------------------------------------------------------
# 1. Determine if the question needs RAG
# --------------------------------------------------------

def is_question_related(question: str) -> bool:
    relevance_prompt = f"""
The user asked: "{question}"

Is this question about the content of their uploaded documents?
Answer only "yes" or "no".
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": relevance_prompt}],
        temperature=0
    )

    ans = resp.choices[0].message.content.strip().lower()
    return ans.startswith("y")  # yes = use RAG


# --------------------------------------------------------
# 2. Vector Store
# --------------------------------------------------------

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )


def retrieve_context(question, k=5):
    # Multi-query expansion
    queries = [question] + generate_queries(question)

    vs = get_vectorstore()
    all_docs = []

    # Collect results from embedding search
    for q in queries:
        all_docs.extend(vs.similarity_search(q, k=k))

    # BM25 keyword search
    bm25 = BM25Store(all_docs)
    all_docs.extend(bm25.search(question, k=k))

    # Remove duplicates
    unique = {d.page_content: d for d in all_docs}.values()
    unique = list(unique)

    # Rerank candidates (this is the magic!)
    pairs = [[question, d.page_content] for d in unique]
    scores = reranker.predict(pairs)

    # Sort by reranker score
    top = sorted(zip(unique, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in top[:k]]

    return top_docs



def generate_queries(question: str):
    prompt = f"""
Generate 3 alternative search queries for RAG retrieval.
Keep them short.

Original question: "{question}"
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    lines = resp.choices[0].message.content.split("\n")
    return [l.strip("-• ") for l in lines if len(l.strip()) > 3]

# --------------------------------------------------------
# 3. Main Answer Function
# --------------------------------------------------------

def answer_question(question: str):
    # STEP A: Check if we should use RAG
    use_rag = is_question_related(question)

    if not use_rag:
        # Answer normally (no context)
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.3
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": []
        }

    # STEP B: If RAG needed → use vector DB
    docs = retrieve_context(question)
    context_text = "\n\n".join(
        f"[{i+1}] {d.metadata.get('source')}\n{d.page_content}"
        for i, d in enumerate(docs)
    )

    rag_prompt = f"""
Use ONLY the following context:

{context_text}

Question: {question}

Answer briefly and include source numbers when relevant.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.1
    )

    # Short, clean source list
    sources = sorted({
        d.metadata.get("source", "unknown").split("/")[-1]
        for d in docs
    })

    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }

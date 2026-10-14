"""
RAG Ingestion Pipeline: Comparison of 8 Chunking Strategies
===========================================================

Run:
    python -m pip install scikit-learn numpy
    python rag.py

This is a small terminal-based demonstration. It uses TF-IDF and cosine
similarity; semantic, LLM-based, and agentic strategies are simplified
offline approximations rather than full production implementations.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. SAMPLE DOCUMENTS
DOCUMENTS = [
    {
        "title": "Leave Policy",
        "department": "HR",
        "section": "Leave",
        "text": (
            "Employees receive 12 casual leave days per year. "
            "Medical leave requires supporting documents. "
            "Leave requests must be approved by the manager."
        ),
    },
    {
        "title": "Attendance Policy",
        "department": "HR",
        "section": "Attendance",
        "text": (
            "Employees must record their daily login and logout time. "
            "Attendance is maintained by the HR department. "
            "Late arrivals must follow company attendance rules."
        ),
    },
    {
        "title": "Salary Policy",
        "department": "Finance",
        "section": "Salary",
        "text": (
            "Employee salaries are processed monthly. "
            "Salary slips contain earnings and deductions. "
            "Salary details are maintained confidentially."
        ),
    },
    {
        "title": "Employee Performance",
        "department": "Management",
        "section": "Performance",
        "text": (
            "Employee performance is evaluated periodically. "
            "Managers review work quality, attendance and goals. "
            "Training can be provided to improve performance."
        ),
    },
]


# 2. TEXT HELPERS
def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalize(text))
        if sentence.strip()
    ]


# 3. CHUNKING STRATEGIES
def fixed_strategy(document, size=20):
    words = document["text"].split()
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), size)
    ]


def recursive_strategy(document, size=30):
    """Simple recursive-style split using sentence and word boundaries."""
    parts = re.split(r"\n\s*\n|(?<=[.!?])\s+", document["text"])
    chunks = []
    current_words = []

    for part in parts:
        words = part.split()

        if current_words and len(current_words) + len(words) > size:
            chunks.append(" ".join(current_words))
            current_words = []

        while len(words) > size:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            chunks.append(" ".join(words[:size]))
            words = words[size:]

        current_words.extend(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def document_strategy(document):
    return [document["text"]]


def semantic_strategy(document):
    # Offline approximation: keep individual sentences as chunks.
    return split_sentences(document["text"])


def query_aware_strategy(document, query=""):
    # Select sentences with the most query-word overlap.
    sentences = split_sentences(document["text"])
    if not query.strip():
        return sentences

    query_terms = set(re.findall(r"\w+", query.lower()))
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            len(query_terms & set(re.findall(r"\w+", item[1].lower()))),
            -item[0],
        ),
        reverse=True,
    )
    return [sentence for _, sentence in ranked[:2]]


def metadata_strategy(document):
    return [
        f"Department: {document['department']}. "
        f"Section: {document['section']}. {document['text']}"
    ]


def llm_based_strategy(document):
    # Offline approximation of topic-aware LLM chunking.
    return split_sentences(document["text"])


def agentic_strategy(document):
    # Offline rule-based approximation of adaptive chunk selection.
    if len(document["text"].split()) > 35:
        return recursive_strategy(document, size=25)
    return split_sentences(document["text"])


STRATEGIES = {
    "Fixed-size": fixed_strategy,
    "Recursive": recursive_strategy,
    "Document-based": document_strategy,
    "Semantic": semantic_strategy,
    "Query-aware": query_aware_strategy,
    "Metadata-aware": metadata_strategy,
    "LLM-based": llm_based_strategy,
    "Agentic": agentic_strategy,
}


# 4. INGESTION PIPELINE
def make_chunk(text, document, index, strategy):
    return {
        "id": f"{strategy}_{index}",
        "text": normalize(text),
        "section": document["section"],
        "department": document["department"],
        "source": document["title"],
        "strategy": strategy,
    }


def ingest_documents(documents=None, strategy="Recursive", query=""):
    documents = documents or DOCUMENTS
    splitter = STRATEGIES[strategy]
    chunks = []

    for document in documents:
        if strategy == "Query-aware":
            pieces = splitter(document, query=query)
        else:
            pieces = splitter(document)

        for piece in pieces:
            if piece.strip():
                chunks.append(
                    make_chunk(piece, document, len(chunks), strategy)
                )

    return chunks


# 5. BUILD A TF-IDF SEARCH INDEX
def create_index(chunks, name="default"):
    if not chunks:
        return {
            "name": name,
            "chunks": [],
            "vectorizer": None,
            "matrix": None,
        }

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    return {
        "name": name,
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


# 6. RETRIEVAL
def search_index(index, query, top_k=3):
    if not query.strip() or not index["chunks"]:
        return []

    query_vector = index["vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, index["matrix"])[0]
    ranked_indices = scores.argsort()[::-1][:top_k]

    return [
        {**index["chunks"][i], "score": float(scores[i])}
        for i in ranked_indices
    ]


# 7. COMPARE ALL EIGHT STRATEGIES
def compare_strategies(query, top_k=3):
    comparisons = {}

    for name in STRATEGIES:
        chunks = ingest_documents(strategy=name, query=query)
        index = create_index(chunks, name)
        comparisons[name] = {
            "chunk_count": len(chunks),
            "results": search_index(index, query, top_k),
        }

    return comparisons


# 8. TERMINAL USER INTERFACE
def main():
    print("=" * 68)
    print("RAG INGESTION PIPELINE: 8 CHUNKING STRATEGIES")
    print("=" * 68)
    print("Sample topics: leave, attendance, salary, performance")
    query = input("\nEnter your question: ").strip()

    if not query:
        print("Please enter a question and run the program again.")
        return

    raw_top_k = input("Number of results per strategy [3]: ").strip()
    try:
        top_k = max(1, min(10, int(raw_top_k))) if raw_top_k else 3
    except ValueError:
        top_k = 3
        print("Invalid number; using top 3 results.")

    comparisons = compare_strategies(query, top_k=top_k)

    print("\n\nRETRIEVAL COMPARISON")
    print("=" * 68)

    for strategy, data in comparisons.items():
        print(f"\nStrategy: {strategy}")
        print(f"Total chunks generated: {data['chunk_count']}")
        print("-" * 68)

        if not data["results"]:
            print("No results found.")
            continue

        for rank, result in enumerate(data["results"], start=1):
            print(f"\nRank: {rank} | Similarity: {result['score']:.3f}")
            print(f"Source: {result['source']}")
            print(f"Department: {result['department']}")
            print(f"Section: {result['section']}")
            print(f"Text: {result['text']}")

    print("\n\nSUMMARY")
    print("=" * 68)
    print(f"{'Strategy':<20} {'Chunks':>8} {'Best score':>12}")
    print("-" * 44)

    for strategy, data in comparisons.items():
        best_score = (
            data["results"][0]["score"] if data["results"] else 0.0
        )
        print(f"{strategy:<20} {data['chunk_count']:>8} {best_score:>12.3f}")

    print("\nNote: TF-IDF similarity is not the same as answer accuracy.")
    print("Semantic, LLM-based, and agentic methods are simplified demos.")


if __name__ == "__main__":
    main()

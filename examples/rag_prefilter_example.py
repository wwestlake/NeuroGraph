"""The flagship use case, kept generic: use NeuroGraph as a cheap,
LLM-free pre-filter in front of a RAG pipeline. Given a large corpus,
build the semantic graph once, then use graph.query() to find the most
relevant existing nodes for an incoming query BEFORE spending anything
on an expensive vector-DB search or an LLM call - only chunks belonging
to the top-ranked nodes need to go further down the pipeline.

This is intentionally not RAG-specific in neurograph itself - graph.py
has no idea this is what it's being used for. That's the point (see
README).

Run: python examples/rag_prefilter_example.py
"""

from neurograph import NeuroGraph

CORPUS = {
    "policy_doc_1": [
        "Employees may carry over up to 5 unused vacation days into the next fiscal year.",
        "Remote work requests must be submitted to a manager at least two weeks in advance.",
    ],
    "policy_doc_2": [
        "All expense reports over $500 require director-level approval.",
        "Reimbursement for travel meals is capped at $75 per day.",
    ],
    "engineering_doc_1": [
        "Pull requests must have at least one approving review before merge.",
        "The staging environment deploys automatically from the main branch.",
    ],
}


def main() -> None:
    graph = NeuroGraph()
    for doc_id, chunks in CORPUS.items():
        graph.ingest_document(doc_id, chunks)

    incoming_query = "how much vacation time can I roll over to next year"
    print(f"Query: {incoming_query!r}\n")

    # Cheap pre-filter: only the top-ranked node(s) need to go on to a
    # real vector search / LLM-backed answer step. Everything else in the
    # corpus is skipped without ever touching an LLM.
    top_nodes = graph.query(incoming_query, top_k=2)
    print("Pre-filtered candidate nodes (would be passed to the next RAG stage):")
    for scored in top_nodes:
        for text in scored.node.member_texts:
            print(f"  score={scored.score:.3f}  \"{text}\"")


if __name__ == "__main__":
    main()

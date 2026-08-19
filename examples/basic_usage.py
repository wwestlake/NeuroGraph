"""Feed a handful of short text chunks spanning a few distinct topics
into NeuroGraph, then inspect the resulting node/edge structure.

Run: python examples/basic_usage.py
"""

from neurograph import NeuroGraph

DOCUMENTS = {
    "doc_astronomy": [
        "The James Webb Space Telescope observes in the infrared spectrum.",
        "Neutron stars form when a massive star's core collapses.",
        "Exoplanets are often detected via the transit method.",
    ],
    "doc_cooking": [
        "Searing meat at high heat creates a Maillard reaction crust.",
        "Sourdough bread relies on wild yeast and lactobacillus cultures.",
        "Braising is a slow, moist-heat cooking technique for tough cuts.",
    ],
    "doc_astronomy_2": [
        "Black holes have an event horizon beyond which light cannot escape.",
        "The transit method also reveals an exoplanet's approximate radius.",
    ],
}


def main() -> None:
    graph = NeuroGraph()

    for doc_id, chunks in DOCUMENTS.items():
        nodes, obs = graph.ingest_document(doc_id, chunks)
        print(f"{doc_id}: {len(chunks)} chunk(s) -> "
              f"{obs.anomaly_count} new node(s), "
              f"{len(nodes) - obs.anomaly_count} joined existing node(s)")

    print(f"\nTotal nodes: {len(graph.nodes)}")
    for node in graph.nodes:
        preview = node.member_texts[0][:60]
        print(f"  [{node.id[:8]}] {len(node)} member(s), "
              f"dispersion={node.dispersion_score():.3f} - \"{preview}...\"")

    print(f"\nTotal edges: {len(graph.edges)}")
    for edge in graph.edges:
        print(f"  {edge.node_a_id[:8]} <-> {edge.node_b_id[:8]}  weight={edge.weight}")

    print("\nQuery: 'how do stars collapse into dense objects'")
    for scored in graph.query("how do stars collapse into dense objects", top_k=3):
        preview = scored.node.member_texts[0][:60]
        print(f"  score={scored.score:.3f}  [{scored.node.id[:8]}] \"{preview}...\"")


if __name__ == "__main__":
    main()

def grounding_check(response: str, source_chunks: list) -> dict:
    matched_sources = []

    for chunk in source_chunks:
        sentences = [
            s.strip() for s in chunk["content"].split(".")
            if len(s.strip()) > 20
        ]
        for sentence in sentences[:3]:
            if sentence.lower() in response.lower():
                matched_sources.append(chunk["source"])
                break

    return {
        "is_grounded":    len(matched_sources) > 0,
        "cited_sources":  list(set(matched_sources))
    }


def detect_hallucination(response: str, source_chunks: list) -> bool:
    source_text = " ".join(
        chunk["content"].lower() for chunk in source_chunks
    )
    source_words = set(source_text.split())

    sentences = [
        s.strip() for s in response.split(".")
        if len(s.strip()) > 10
    ]

    unsupported = []
    for sentence in sentences:
        words = set(sentence.lower().split())
        overlap = words & source_words
        overlap_ratio = len(overlap) / max(len(words), 1)
        if overlap_ratio < 0.2:
            unsupported.append(sentence)

    hallucination_ratio = len(unsupported) / max(len(sentences), 1)
    return hallucination_ratio > 0.3


def safe_response(
    query: str,
    llm_response: str,
    source_chunks: list
) -> dict:
    grounding  = grounding_check(llm_response, source_chunks)
    hallucinated = detect_hallucination(llm_response, source_chunks)

    final_response = (
        llm_response
        if grounding["is_grounded"]
        else "I could not find a reliable answer in the provided context."
    )

    return {
        "query":             query,
        "response":          final_response,
        "is_grounded":       grounding["is_grounded"],
        "cited_sources":     grounding["cited_sources"],
        "hallucination_risk": hallucinated
    }


if __name__ == "__main__":
    sample_chunks = [
        {
            "content": "The coffee automaton is a two-dimensional array of bits describing the system state",
            "source":  "Quanitify.pdf"
        }
    ]

    # Test 1 — grounded response
    grounded = "The coffee automaton is a two-dimensional array of bits describing the system state with ones for cream and zeros for coffee"
    result = safe_response("What is the coffee automaton?", grounded, sample_chunks)
    print("Test 1 — Grounded response:")
    print(f"  is_grounded      : {result['is_grounded']}")
    print(f"  hallucination    : {result['hallucination_risk']}")
    print(f"  cited_sources    : {result['cited_sources']}")

    print()

    # Test 2 — hallucinated response
    hallucinated = "The coffee automaton was invented in 1995 by a team at MIT and won the Nobel Prize in Physics"
    result = safe_response("What is the coffee automaton?", hallucinated, sample_chunks)
    print("Test 2 — Hallucinated response:")
    print(f"  is_grounded      : {result['is_grounded']}")
    print(f"  hallucination    : {result['hallucination_risk']}")
    print(f"  response         : {result['response']}")
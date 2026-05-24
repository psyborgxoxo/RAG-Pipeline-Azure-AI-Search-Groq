import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(__file__))
from query import hybrid_search

def evaluate_retrieval(test_cases: list) -> None:
    print("=" * 60)
    print("RAG Retrieval Evaluation")
    print("=" * 60)

    recall_scores = []
    mrr_scores    = []

    for case in test_cases:
        query    = case["query"]
        relevant = case["relevant_keywords"]

        results  = hybrid_search(query, top_k=5)
        retrieved_contents = [r["content"].lower() for r in results]

        # R@5 — how many relevant keywords found in top 5
        found = sum(
            1 for kw in relevant
            if any(kw.lower() in content for content in retrieved_contents)
        )
        recall = found / len(relevant)
        recall_scores.append(recall)

        # MRR — rank of first relevant result
        mrr = 0.0
        for i, content in enumerate(retrieved_contents):
            if any(kw.lower() in content for kw in relevant):
                mrr = 1.0 / (i + 1)
                break
        mrr_scores.append(mrr)

        print(f"\nQuery   : {query}")
        print(f"R@5     : {recall:.2f}")
        print(f"MRR     : {mrr:.2f}")
        print(f"Top result preview: {results[0]['content'][:150]}...")

    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_mrr    = sum(mrr_scores)    / len(mrr_scores)

    print("\n" + "=" * 60)
    print(f"Average R@5 : {avg_recall:.2f}")
    print(f"Average MRR : {avg_mrr:.2f}")
    print("=" * 60)

    if avg_recall >= 0.8:
        print("Retrieval quality : GOOD")
    elif avg_recall >= 0.5:
        print("Retrieval quality : ACCEPTABLE")
    else:
        print("Retrieval quality : NEEDS IMPROVEMENT")


if __name__ == "__main__":
    test_cases = [
        {
            "query":            "What is the coffee automaton?",
            "relevant_keywords": ["cellular automaton", "coffee", "cream", "bits"]
        },
        {
            "query":            "How is complexity measured?",
            "relevant_keywords": ["complexity", "measure", "entropy"]
        },
        {
            "query":            "What happens to complexity over time?",
            "relevant_keywords": ["increases", "decreases", "time", "closed systems"]
        }
    ]

    evaluate_retrieval(test_cases)
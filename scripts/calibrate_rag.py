import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tripmate.config import get_settings
from tripmate.rag.store import DestinationStore


IN_SCOPE = [
    "do I need a visa",
    "tipping customs",
    "pickpocket risk",
    "what to wear at temples",
    "northern lights season",
    "is tap water safe",
    "best month to visit",
    "packing tips",
]

OUT_OF_SCOPE = [
    "capital of France",
    "python sorting algorithm",
    "book a flight",
    "stock market advice",
    "how to bake bread",
    "who won the world cup",
]


def main():
    settings = get_settings()
    vector_store = DestinationStore(settings)._vs

    def best_distance(query):
        results = vector_store.similarity_search_with_score(
            query, k=1
        )
        if not results:
            raise RuntimeError(
                f"No matching documents found for: {query}"
            )
        return results[0][1]

    print("IN SCOPE")
    in_scores = []
    for query in IN_SCOPE:
        score = best_distance(query)
        in_scores.append(score)
        print(f"  {score:.3f}  {query}")

    print("\nOUT OF SCOPE")
    out_scores = []
    for query in OUT_OF_SCOPE:
        score = best_distance(query)
        out_scores.append(score)
        print(f"  {score:.3f}  {query}")

    print("\nSUMMARY")
    print(f"IN SCOPE max distance:  {max(in_scores):.3f}")
    print(f"OUT OF SCOPE min distance: {min(out_scores):.3f}")

    if max(in_scores) < min(out_scores):
        print("The groups are separated.")
        print(
            "A threshold between these two values "
            "separates this test set."
        )
    else:
        print("The groups overlap.")
        print(
            "No single threshold perfectly separates "
            "these test queries."
        )


if __name__ == "__main__":
    main()

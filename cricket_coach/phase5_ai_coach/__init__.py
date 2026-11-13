"""Retrieval-assisted coaching and optional voice feedback."""

if __name__ == "__main__":
    from cricket_coach.phase5_ai_coach.rag_coach import RAGCoach

    assert RAGCoach().diagnose({})["observations"]
    print("Coaching package checks passed.")

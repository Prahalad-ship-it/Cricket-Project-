"""Standalone coaching package checks."""

from cricket_coach.phase5_ai_coach.rag_coach import RAGCoach


if __name__ == "__main__":
    assert RAGCoach().diagnose({})["observations"]
    print("Coaching package checks passed.")

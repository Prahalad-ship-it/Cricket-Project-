"""Coaching observations from local guidance and optional retrieval/LLM providers."""

from pathlib import Path
from typing import Any, Callable, Iterable

from cricket_coach.data_contract import load_phase_payload, save_phase_payload


class RAGCoach:
    """Generate bounded, transparent coaching notes from measured findings."""

    def __init__(
        self,
        documents: Iterable[str] = (),
        retriever: Callable[[str], list[str]] | None = None,
        llm: Callable[[str, list[str]], str] | None = None,
    ) -> None:
        self.documents = [document.strip() for document in documents if document.strip()]
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def from_text_directory(cls, directory: str | Path, **kwargs: Any) -> "RAGCoach":
        """Create a simple local text retriever from .txt/.md manuals."""
        root = Path(directory)
        documents = [path.read_text(encoding="utf-8") for path in root.glob("*.txt")]
        documents.extend(path.read_text(encoding="utf-8") for path in root.glob("*.md"))
        return cls(documents, **kwargs)

    @classmethod
    def from_chroma_collection(cls, collection: Any, llm: Callable[[str, list[str]], str] | None = None) -> "RAGCoach":
        """Use an existing ChromaDB collection as a retriever without owning its lifecycle."""
        def retrieve(query: str) -> list[str]:
            response = collection.query(query_texts=[query], n_results=3)
            return list(response.get("documents", [[]])[0] or [])

        return cls(retriever=retrieve, llm=llm)

    def diagnose(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Write observations grounded only in available pipeline metrics."""
        observations: list[str] = []
        biomechanics = metadata.get("biomechanics", {})
        if biomechanics.get("knee_brace_warning") is True:
            observations.append("The estimated front-knee angle was below the configured screening threshold; review the landing frame.")
        if biomechanics.get("trunk_tilt_warning") is True:
            observations.append("The estimated trunk tilt crossed the configured screening threshold; review camera alignment and technique.")
        if biomechanics.get("elbow_extension_screen") is True:
            observations.append("The estimated elbow extension crossed a screening threshold; this is not a legality decision.")
        if not observations:
            observations.append("No configured screening threshold was conclusively crossed; review the clip and confirm the pose estimate.")
        query = " ".join(observations)
        if self.retriever:
            context = self.retriever(query)
        else:
            context = self._retrieve(query)
        if self.llm:
            feedback = self.llm(query, context)
        else:
            feedback = " ".join(observations)
        result = {
            "feedback": feedback,
            "observations": observations,
            "retrieved_context": context,
            "generated_by": "configured_llm" if self.llm else "deterministic_local_rules",
            "safety_note": "Automated video estimates are coaching aids, not medical advice or an official bowling-action legality assessment.",
        }
        return result

    def _retrieve(self, query: str) -> list[str]:
        words = set(query.lower().split())
        ranked = sorted(
            self.documents,
            key=lambda document: len(words.intersection(document.lower().split())),
            reverse=True,
        )
        return ranked[:3]

    def process(self, metadata_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
        metadata, arrays = load_phase_payload(metadata_path)
        metadata["coach"] = self.diagnose(metadata)
        metadata["phase_status"] = {**metadata.get("phase_status", {}), "coach": "complete"}
        save_phase_payload(metadata_path, metadata, arrays)
        return metadata, arrays


if __name__ == "__main__":
    result = RAGCoach().diagnose({"biomechanics": {}})
    assert result["generated_by"] == "deterministic_local_rules"
    print("Coaching checks passed.")

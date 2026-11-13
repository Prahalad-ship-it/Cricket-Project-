"""Coaching observations from local guidance and optional retrieval/LLM providers."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from cricket_coach.config import (
    ELBOW_EXTENSION_WARNING_DEG,
    NVIDIA_API_BASE_URL,
    NVIDIA_COACH_MAX_TOKENS,
    NVIDIA_COACH_MODEL,
    NVIDIA_COACH_TEMPERATURE,
    NVIDIA_COACH_TOP_P,
)
from cricket_coach.data_contract import load_phase_payload, save_phase_payload

LOGGER = logging.getLogger(__name__)

COACH_SYSTEM_PROMPT = """You are an elite international fast-bowling coach reviewing a video-derived bowling report.
Use only the supplied measurements and context. Clearly distinguish measured values from uncertain estimates.
Explain how relevant joint mechanics may affect pace generation, bowling-action legality screening, or lower-back loading.
An elbow extension estimate above 15 degrees is a screening concern only: never declare an action legal or illegal from this report.
Hip-shoulder separation is not a diagnosis or a validated prediction of injury; avoid medical certainty.
Return only valid JSON with exactly these string keys:
{
  "root_cause_diagnosis": "Explain the most important supported mechanical cause and its implications.",
  "priority_fix": "Give the single most impactful adjustment, in one concise sentence.",
  "prescribed_net_drill": "Give one concrete, repeatable drill for the very next delivery."
}
If the evidence is unavailable or inconclusive, say so plainly and prescribe a safe observation-focused drill instead of inventing findings."""


class RAGCoach:
    """Generate rule-based or NVIDIA NIM coaching from structured delivery metrics."""

    def __init__(
        self,
        documents: Iterable[str] = (),
        retriever: Callable[[str], list[str]] | None = None,
        llm: Callable[[str, list[str]], str] | None = None,
        client: Any = None,
    ) -> None:
        self.documents = [document.strip() for document in documents if document.strip()]
        self.retriever = retriever
        self.llm = llm
        self.nvidia_client = client if client is not None else self._create_nvidia_client()

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

    @staticmethod
    def _create_nvidia_client() -> Any:
        """Create an OpenAI-compatible NVIDIA client only when a key is configured."""
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError as error:
            LOGGER.warning("NVIDIA coaching disabled because OpenAI SDK is unavailable: %s", error)
            return None
        return OpenAI(
            base_url=NVIDIA_API_BASE_URL,
            api_key=api_key,
        )

    def diagnose(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Create a structured, evidence-bounded coaching report."""
        observations: list[str] = []
        biomechanics = metadata.get("biomechanics", {})
        injury_screen = metadata.get("injury_screen", {})
        event_detection = metadata.get("event_detection", metadata.get("events", {}))
        knee = biomechanics.get("front_knee_angle_deg", biomechanics.get("knee_angle_deg", {}))
        elbow = biomechanics.get("bowling_elbow_angle_deg", biomechanics.get("elbow_angle_deg", {}))
        trunk = biomechanics.get("lateral_trunk_tilt_deg", biomechanics.get("trunk_tilt_deg", {}))
        if knee.get("min") is not None:
            observations.append(f"Minimum estimated front-knee angle in the analysis window: {knee['min']:.1f} degrees.")
        if elbow.get("min") is not None:
            observations.append(f"Minimum estimated bowling-elbow included angle: {elbow['min']:.1f} degrees.")
        extension = biomechanics.get("bowling_elbow_extension_deg", {})
        if extension.get("max") is not None:
            observations.append(f"Maximum estimated elbow extension: {extension['max']:.1f} degrees.")
        if trunk.get("max") is not None:
            observations.append(f"Maximum estimated lateral trunk tilt: {trunk['max']:.1f} degrees.")
        if biomechanics.get("knee_brace_warning") is True:
            observations.append("The estimated front-knee angle was below the configured screening threshold; review the landing frame.")
        if biomechanics.get("trunk_tilt_warning") is True:
            observations.append("The estimated trunk tilt crossed the configured screening threshold; review camera alignment and technique.")
        if biomechanics.get("elbow_extension_screen") is True:
            observations.append(
                f"The estimated elbow extension crossed the {ELBOW_EXTENSION_WARNING_DEG:.0f}-degree screening threshold; this is not a legality decision."
            )
        if injury_screen.get("separation_warning") is True:
            observations.append("Hip-shoulder separation crossed the configured screening threshold; this is not an injury prediction.")
        if not observations:
            observations.append("Biomechanical values are unavailable or no configured screening threshold was conclusively crossed; review the video and validate the pose estimate.")

        structured_input = {
            "biomechanics": biomechanics,
            "injury_screen": injury_screen,
            "event_detection": event_detection,
            "rule_based_observations": observations,
        }
        query = json.dumps(structured_input, indent=2, allow_nan=False, default=self._json_default)
        if self.retriever:
            context = self.retriever(" ".join(observations))
        else:
            context = self._retrieve(" ".join(observations))

        if self.llm:
            try:
                feedback = self.llm(query, context)
            except Exception as error:
                LOGGER.warning("Configured coaching provider failed (%s); using rule-based coaching.", type(error).__name__)
                feedback = None
            generated_by = "configured_llm"
        elif self.nvidia_client is not None:
            try:
                response = self.nvidia_client.chat.completions.create(
                    model=NVIDIA_COACH_MODEL,
                    messages=[
                        {"role": "system", "content": COACH_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Analyze this structured bowling delivery report and return the requested coaching JSON.\n\n"
                                f"{query}\n\nRetrieved coaching context:\n{json.dumps(context, ensure_ascii=False)}"
                            ),
                        },
                    ],
                    temperature=NVIDIA_COACH_TEMPERATURE,
                    top_p=NVIDIA_COACH_TOP_P,
                    max_tokens=NVIDIA_COACH_MAX_TOKENS,
                    stream=False,
                )
                feedback = response.choices[0].message.content
                generated_by = NVIDIA_COACH_MODEL
            except Exception as error:
                LOGGER.warning("NVIDIA coaching request failed (%s); using rule-based coaching.", type(error).__name__)
                feedback = None
                generated_by = "deterministic_local_rules"
        else:
            feedback = None
            generated_by = "deterministic_local_rules"

        if isinstance(feedback, str) and feedback.strip():
            report = self._parse_report(feedback.strip())
            if report is None:
                LOGGER.warning("Coaching provider returned an invalid report; using rule-based coaching.")
                report = self._fallback_report(observations)
                generated_by = "deterministic_local_rules"
        else:
            report = self._fallback_report(observations)
            generated_by = "deterministic_local_rules"

        result = {
            "feedback": report,
            "report": report,
            "observations": observations,
            "retrieved_context": context,
            "generated_by": generated_by,
            "safety_note": "Automated video estimates are coaching aids, not medical advice or an official bowling-action legality assessment.",
        }
        return result

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")

    @staticmethod
    def _parse_report(content: str) -> dict[str, str] | None:
        """Accept JSON only when it contains all required coaching sections."""
        candidate = content.strip()
        if candidate.startswith("```"):
            candidate = candidate.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            report = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        required = ("root_cause_diagnosis", "priority_fix", "prescribed_net_drill")
        if not isinstance(report, dict) or any(not isinstance(report.get(key), str) or not report[key].strip() for key in required):
            return None
        return {key: report[key].strip() for key in required}

    @staticmethod
    def _fallback_report(observations: list[str]) -> dict[str, str]:
        """Return safe actionable coaching when the external service is unavailable."""
        root_cause = " ".join(observations)
        return {
            "root_cause_diagnosis": root_cause,
            "priority_fix": "Prioritize a repeatable, balanced action and review the delivery from the same camera angle before changing mechanics.",
            "prescribed_net_drill": "For the next six deliveries, bowl at comfortable effort while a partner records side-on video; review front-knee position, trunk alignment, and the release frame after each ball.",
        }

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

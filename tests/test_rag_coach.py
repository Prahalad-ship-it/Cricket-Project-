import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from cricket_coach.phase5_ai_coach.rag_coach import RAGCoach


class RAGCoachFallbackTests(unittest.TestCase):
    def test_empty_nvidia_report_uses_local_coaching(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        return_value=SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content="")
                                )
                            ]
                        )
                    )
                )
            )
        )

        result = RAGCoach(client=client).diagnose({"biomechanics": {}})

        self.assertEqual(result["generated_by"], "deterministic_local_rules")
        self.assertTrue(result["report"]["priority_fix"])
        self.assertTrue(result["report"]["prescribed_net_drill"])

    def test_invalid_nvidia_report_uses_local_coaching(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(
                        return_value=SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content="not JSON")
                                )
                            ]
                        )
                    )
                )
            )
        )

        result = RAGCoach(client=client).diagnose({"biomechanics": {}})

        self.assertEqual(result["generated_by"], "deterministic_local_rules")
        self.assertTrue(result["report"]["priority_fix"])
        self.assertTrue(result["report"]["prescribed_net_drill"])


if __name__ == "__main__":
    unittest.main()

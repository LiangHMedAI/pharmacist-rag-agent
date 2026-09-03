from types import SimpleNamespace
import unittest

from interaction_rules import find_interaction
from llm_service import (
    SAFETY_NOTICE,
    no_evidence_response,
    page_number,
    source_name,
    validate_grounded_response,
)


def make_doc(source="C:\\private\\Aspirin_Insert.pdf", page=0):
    return SimpleNamespace(metadata={"source": source, "page": page})


class SafetyTests(unittest.TestCase):
    def test_source_name_never_leaks_local_path(self):
        self.assertEqual(source_name(make_doc()), "Aspirin_Insert.pdf")
        self.assertEqual(
            source_name(make_doc("/srv/private/Ibuprofen.pdf")), "Ibuprofen.pdf"
        )

    def test_page_number_is_human_friendly(self):
        self.assertEqual(page_number(make_doc(page=0)), 1)
        self.assertEqual(page_number(make_doc(page=4)), 5)

    def test_claim_without_citation_fails_closed(self):
        answer = validate_grounded_response(
            "【EVIDENCE FOUND】\nA medical claim without a citation."
        )
        self.assertTrue(answer.startswith("【NOT FOUND IN PROVIDED SOURCES】"))

    def test_cited_answer_passes_and_gets_safety_notice(self):
        answer = validate_grounded_response(
            "【EVIDENCE FOUND】\nThe excerpt contains this fact. [Source 1, page 2]"
        )
        self.assertTrue(answer.startswith("【EVIDENCE FOUND】"))
        self.assertIn(SAFETY_NOTICE, answer)

    def test_no_evidence_response_is_conservative(self):
        answer = no_evidence_response("No relevant source.")
        self.assertTrue(answer.startswith("【NOT FOUND IN PROVIDED SOURCES】"))
        self.assertIn("No medication recommendation", answer)

    def test_interaction_lookup_is_order_independent(self):
        self.assertIsNotNone(find_interaction("Warfarin", "Amoxicillin"))
        self.assertIsNotNone(find_interaction("Amoxicillin", "Warfarin"))

    def test_missing_interaction_rule_does_not_claim_safety(self):
        self.assertIsNone(find_interaction("Acetaminophen", "Metformin"))


if __name__ == "__main__":
    unittest.main()

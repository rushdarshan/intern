import unittest

from graph import classify_question


class GraphRoutingTests(unittest.TestCase):
    def test_domain_question_without_company_name_routes_to_debales(self):
        result = classify_question({"question": "What integrations do you support?"})
        self.assertEqual(result["route"], "debales")

    def test_mixed_question_routes_to_both(self):
        result = classify_question({"question": "Compare Debales AI with OpenAI and tell me OpenAI's latest model."})
        self.assertEqual(result["route"], "both")

    def test_short_question_routes_to_unknown(self):
        result = classify_question({"question": "hi"})
        self.assertEqual(result["route"], "unknown")


if __name__ == "__main__":
    unittest.main()

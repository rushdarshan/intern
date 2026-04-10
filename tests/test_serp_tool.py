import os
import unittest
from unittest.mock import Mock, patch

from serp_tool import serp_search


class SerpToolTests(unittest.TestCase):
    @patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"}, clear=False)
    @patch("serp_tool.requests.get")
    def test_serp_search_collects_direct_answers_and_organic_results(self, mock_get):
        response = Mock()
        response.json.return_value = {
            "answer_box": {"title": "Paris", "answer": "Paris", "link": "https://example.com/paris"},
            "organic_results": [
                {
                    "title": "Capital of France",
                    "link": "https://example.com/france",
                    "snippet": "Paris is the capital of France.",
                }
            ],
        }
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        result = serp_search("capital of france")

        self.assertIsNone(result["error"])
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["title"], "Paris")


if __name__ == "__main__":
    unittest.main()

import unittest

from scraper import _is_allowed_url, _normalize_url


class ScraperTests(unittest.TestCase):
    def test_normalize_url_strips_query_fragment_and_trailing_slash(self):
        self.assertEqual(
            _normalize_url("https://debales.ai/blog/example/?utm=1#section"),
            "https://debales.ai/blog/example",
        )

    def test_allowed_url_keeps_relevant_sections(self):
        self.assertTrue(_is_allowed_url("https://debales.ai/integrations"))
        self.assertTrue(_is_allowed_url("https://debales.ai/ai-agents"))

    def test_disallowed_url_blocks_auth_pages(self):
        self.assertFalse(_is_allowed_url("https://debales.ai/sign-in"))


if __name__ == "__main__":
    unittest.main()

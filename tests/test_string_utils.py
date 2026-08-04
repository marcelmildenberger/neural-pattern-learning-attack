import string
import unittest

from utils.string_utils import (
    extract_bi_grams,
    get_all_bi_grams,
    normalize_for_bigrams,
)


class BigramStringUtilsTests(unittest.TestCase):
    def test_vocabulary_contains_every_ordered_alphanumeric_pair(self):
        alphabet = string.ascii_lowercase + string.digits
        expected = [first + second for first in alphabet for second in alphabet]

        vocabulary = get_all_bi_grams()

        self.assertEqual(vocabulary, expected)
        self.assertEqual(len(vocabulary), len(alphabet) ** 2)
        self.assertIn("a1", vocabulary)
        self.assertIn("1a", vocabulary)

    def test_tokenization_uses_lowercase_alphanumeric_normalization(self):
        value = 'A1 B-2/C."'

        self.assertEqual(normalize_for_bigrams(value), "a1b2c")
        self.assertEqual(extract_bi_grams(value), ["a1", "1b", "b2", "2c"])

    def test_remove_spaces_compatibility_argument_does_not_change_output(self):
        value = "ab cd"

        self.assertEqual(
            extract_bi_grams(value, remove_spaces=False),
            extract_bi_grams(value, remove_spaces=True),
        )


if __name__ == "__main__":
    unittest.main()

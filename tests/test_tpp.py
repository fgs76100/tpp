import unittest
from tpp.tpp import tpp_preprocessor


class TPP_preprocess(unittest.TestCase):
    def test_tpp_preprocessor(self):
        self.assertEqual(tpp_preprocessor(r"${foo}"), r"${'${'}foo${'}'}")
        self.assertEqual(tpp_preprocessor(r"foo \bar"), r"foo ${'\\'}bar")
        self.assertEqual(tpp_preprocessor(r"@{{bar}}"), r"${bar}")

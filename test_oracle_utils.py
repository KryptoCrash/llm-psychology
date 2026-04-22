import unittest

import torch
from transformers.tokenization_utils_base import BatchEncoding

import oracle_utils


class DummyTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return BatchEncoding({"input_ids": [[101, 7, 7, 102]]})

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == oracle_utils.SPECIAL_TOKEN:
            return [7]
        if text == "Answer:":
            return [201, 202]
        raise AssertionError(f"Unexpected text: {text!r}")


class OracleUtilsTest(unittest.TestCase):
    def test_normalize_token_ids_accepts_batch_encoding(self):
        encoded = BatchEncoding({"input_ids": [[11, 12, 13]]})
        self.assertEqual(oracle_utils._normalize_token_ids(encoded), [11, 12, 13])

    def test_normalize_token_ids_accepts_tensor(self):
        token_ids = torch.tensor([[21, 22, 23]])
        self.assertEqual(oracle_utils._normalize_token_ids(token_ids), [21, 22, 23])

    def test_create_oracle_input_accepts_chat_template_batch_encoding(self):
        tokenizer = DummyTokenizer()
        oracle_input = oracle_utils.create_oracle_input(
            prompt="Why?",
            layer=3,
            num_positions=2,
            tokenizer=tokenizer,
            acts_BD=torch.randn(2, 4),
            forced_model_prefix="Answer:",
        )

        self.assertEqual(oracle_input.input_ids, [101, 7, 7, 102, 201, 202])
        self.assertEqual(oracle_input.positions, [1, 2])


if __name__ == "__main__":
    unittest.main()

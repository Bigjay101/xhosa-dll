
"""T9: MLM scoring correctness tests (handbook D3)."""

import pytest
import torch
import torch.nn.functional as F

from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM

from src.mlm_scoring import build_masks, score_mlm
from src.segment import words


class ToyEncoding(dict):
    """Minimal tokenizer output supporting word_ids()."""

    def __init__(self, ids, word_ids):
        super().__init__(
            input_ids=torch.tensor([ids], dtype=torch.long),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
        )
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        assert batch_index == 0
        return self._word_ids


class ToyTokenizer:
    """Fixed word-to-subword mapping; no external downloads."""

    is_fast = True
    mask_token_id = 3
    model_max_length = 64

    pieces = {
        "aa": [4, 5],
        "bb": [6],
        "cc": [7, 8],
    }

    def __call__(
        self, ws, is_split_into_words=True,
        return_tensors="pt", truncation=False
    ):
        assert is_split_into_words
        assert return_tensors == "pt"

        ids = [0]  # BOS
        word_ids = [None]

        for index, word in enumerate(ws):
            tokens = self.pieces[word]
            ids.extend(tokens)
            word_ids.extend([index] * len(tokens))

        ids.append(2)  # EOS
        word_ids.append(None)

        return ToyEncoding(ids, word_ids)


@pytest.fixture
def setup_mlm():
    torch.manual_seed(13)

    config = XLMRobertaConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )

    model = XLMRobertaForMaskedLM(config).eval()
    tokenizer = ToyTokenizer()

    return model, tokenizer


def test_batch_size_invariance(setup_mlm):
    model, tokenizer = setup_mlm
    texts = ["aa bb cc", "cc aa"]

    small = score_mlm(
        model, tokenizer, texts,
        batch_size=1, device="cpu"
    )

    large = score_mlm(
        model, tokenizer, texts,
        batch_size=8, device="cpu"
    )

    for a, b in zip(small, large):
        assert a["total_nll"] == pytest.approx(
            b["total_nll"], abs=1e-5
        )
        assert a["word_nll"] == pytest.approx(
            b["word_nll"], abs=1e-5
        )


def test_fast_path_matches_full_model(setup_mlm):
    model, tokenizer = setup_mlm
    text = "aa bb cc"

    fast = score_mlm(
        model, tokenizer, [text],
        batch_size=8, device="cpu"
    )[0]

    enc = tokenizer(
        words(text),
        is_split_into_words=True,
        return_tensors="pt",
        truncation=False,
    )

    word_ids = enc.word_ids()
    masks = build_masks(word_ids, method="l2r")
    positions = [
        i for i, w in enumerate(word_ids) if w is not None
    ]

    reference_nll = [0.0] * 3

    with torch.no_grad():
        for target_pos, masked_positions in zip(positions, masks):
            masked_ids = enc["input_ids"].clone()
            masked_ids[0, masked_positions] = tokenizer.mask_token_id

            # Standard full-vocabulary forward pass.
            logits = model(
                input_ids=masked_ids,
                attention_mask=enc["attention_mask"],
            ).logits[0, target_pos]

            true_token = enc["input_ids"][0, target_pos]

            loss = F.cross_entropy(
                logits.unsqueeze(0),
                true_token.reshape(1),
            ).item()

            reference_nll[word_ids[target_pos]] += loss

    assert fast["word_nll"] == pytest.approx(
        reference_nll, abs=1e-5
    )


def test_word_nll_consistency(setup_mlm):
    model, tokenizer = setup_mlm

    result = score_mlm(
        model, tokenizer, ["aa bb cc"],
        device="cpu",
    )[0]

    assert result["n_words"] == 3
    assert result["n_tokens"] == 5
    assert result["word_pieces"] == [2, 1, 2]
    assert len(result["word_nll"]) == 3

    assert sum(result["word_nll"]) == pytest.approx(
        result["total_nll"], abs=1e-6
    )
    assert all(loss >= 0 for loss in result["word_nll"])

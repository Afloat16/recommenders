# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import json

import numpy as np
import pandas as pd
import pytest
from transformers import BertTokenizer

from recommenders.models.tfidf.tfidf_utils import TfidfRecommender


@pytest.fixture
def local_bert_tokenizer(tmp_path, monkeypatch):
    # Exercise the real WordPiece tokenizer with a small local vocabulary; only
    # the pretrained-vocabulary download is replaced for this offline regression.
    (tmp_path / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nplay\n##ing\ncello\nmusic\ntheory\n",
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "bert"}), encoding="utf-8"
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"do_lower_case": False, "tokenizer_class": "BertTokenizer"}),
        encoding="utf-8",
    )
    tokenizer = BertTokenizer.from_pretrained(str(tmp_path))
    monkeypatch.setattr(BertTokenizer, "from_pretrained", lambda _: tokenizer)
    return tokenizer


@pytest.mark.parametrize("method", ["bert", "scibert"])
@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(3),
        pd.Index([2, 0, 1]),
        pd.Index([10, 20, 30]),
        pd.Index([0, 0, 1]),
        pd.Index(["doc-c", "doc-a", "doc-b"]),
        pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("b", 3)]),
    ],
    ids=["range", "reordered", "nonzero", "duplicates", "strings", "multiindex"],
)
def test_bert_tokenization_preserves_dataframe_index(
    index, method, local_bert_tokenizer
):
    frame = pd.DataFrame(
        {
            "itemID": ["A", "B", "C"],
            "cleaned_text": ["playing cello", "music theory", "cello music"],
        },
        index=index,
    )
    original = frame.copy(deep=True)
    model = TfidfRecommender(id_col="itemID", tokenization_method=method)
    vectorizer, texts = model.tokenize_text(frame, ngram_range=(1, 1), min_df=1)

    expected = pd.Series(
        ["play ##ing cello", "music theory", "cello music"],
        index=index,
        name="cleaned_text",
    )
    pd.testing.assert_series_equal(texts, expected)
    pd.testing.assert_frame_equal(frame, original)
    model.fit(vectorizer, texts)

    reference_frame = frame.reset_index(drop=True)
    reference = TfidfRecommender(id_col="itemID", tokenization_method=method)
    reference_vectorizer, reference_texts = reference.tokenize_text(
        reference_frame, ngram_range=(1, 1), min_df=1
    )
    reference.fit(reference_vectorizer, reference_texts)
    np.testing.assert_allclose(
        model.tfidf_matrix.toarray(), reference.tfidf_matrix.toarray()
    )
    pd.testing.assert_frame_equal(
        model.recommend_top_k_items(frame, k=1),
        reference.recommend_top_k_items(reference_frame, k=1),
    )

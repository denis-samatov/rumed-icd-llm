"""Method 0: TF-IDF (word + char n-grams) + logistic regression.

A sanity check for the harness: it should land near the paper's feature-based baseline
(Hit@1 49.76 / Hit@3 72.75 on test, arXiv:2201.06499)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from rumed_icd.data import Record


class TfidfBaseline:
    def __init__(self, c: float = 10.0, seed: int = 0) -> None:
        features = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                     sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                     sublinear_tf=True, lowercase=True)),
        ])
        self.model = make_pipeline(
            features, LogisticRegression(C=c, max_iter=2000, random_state=seed)
        )

    def fit(self, train: Sequence[Record]) -> TfidfBaseline:
        self.model.fit([r.text for r in train], [r.code for r in train])
        return self

    def predict_topk(self, records: Sequence[Record], k: int = 3) -> list[list[str]]:
        proba = self.model.predict_proba([r.text for r in records])
        classes = self.model.classes_
        order = np.argsort(-proba, axis=1)[:, :k]
        return [[str(classes[j]) for j in row] for row in order]

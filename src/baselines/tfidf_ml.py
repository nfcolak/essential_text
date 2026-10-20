"""Tier 2 TF-IDF + Classical ML Baselines (Logistic Regression and Linear SVM).

Implements:
- Strict train-only vocabulary fitting:
  TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=25000)
- Hyperparameter tuning over C in {0.01, 0.1, 1.0, 10.0} and class_weight in [None, "balanced"]
  maximizing validation Macro-F1
- Calibrated 3-class probability outputs for both Logistic Regression and Linear SVM
  (via Platt/sigmoid scaling) satisfying the probability simplex (rows sum to 1.0)
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

LABEL_TO_ID: dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL: dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def _compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Macro-F1 across classes 0, 1, 2."""
    f1s: list[float] = []
    for c in (0, 1, 2):
        tp = int(np.count_nonzero((y_true == c) & (y_pred == c)))
        fp = int(np.count_nonzero((y_true != c) & (y_pred == c)))
        fn = int(np.count_nonzero((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


class TfidfMLBaseline:
    """Tier 2 TF-IDF ML baseline supporting Logistic Regression and Linear SVM.

    Parameters
    ----------
    model_type : str
        Either "logreg" (Logistic Regression) or "svm" (Linear SVM with Platt calibration).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, model_type: str = "logreg", random_state: int = 42) -> None:
        if model_type not in {"logreg", "svm"}:
            raise ValueError(f"model_type must be 'logreg' or 'svm', got '{model_type}'")
        self.model_type: str = model_type
        self.random_state: int = random_state

        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=25000,
        )
        self.classifier_: Any = None
        self.best_params_: dict[str, Any] = {
            "model_type": self.model_type,
            "C": 1.0,
            "class_weight": None,
            "val_macro_f1": None,
        }

    def _build_and_fit_classifier(
        self,
        c_val: float,
        class_weight: Any,
        x_train: Any,
        y_train: np.ndarray,
    ) -> Any:
        """Instantiate and train candidate classifier on training features."""
        if self.model_type == "logreg":
            clf = LogisticRegression(
                C=c_val,
                class_weight=class_weight,
                max_iter=1000,
                random_state=self.random_state,
            )
            clf.fit(x_train, y_train)
            return clf
        elif self.model_type == "svm":
            base_svm = LinearSVC(
                C=c_val,
                class_weight=class_weight,
                random_state=self.random_state,
                max_iter=2000,
            )
            base_svm.fit(x_train, y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                calibrated = CalibratedClassifierCV(
                    estimator=base_svm,
                    method="sigmoid",
                    cv="prefit",
                )
                calibrated.fit(x_train, y_train)
            return calibrated
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str] | None = None,
        val_labels: list[int] | None = None,
    ) -> TfidfMLBaseline:
        """Fit vectorizer strictly on train_texts and tune hyperparameters on validation set.

        Sweeps C in {0.01, 0.1, 1.0, 10.0} and class_weight in [None, "balanced"] to maximize
        validation Macro-F1.
        """
        # Strictly fit vectorizer ONLY on train_texts
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=25000,
        )
        x_train = self.vectorizer.fit_transform(train_texts)
        y_train = np.asarray(train_labels, dtype=int)

        target_val_texts = val_texts if (val_texts is not None and len(val_texts) > 0) else None
        target_val_labels = val_labels if (val_labels is not None and len(val_labels) > 0) else None

        if target_val_texts is not None and target_val_labels is not None:
            # Transform validation set using vocabulary learned strictly from training set
            x_val = self.vectorizer.transform(target_val_texts)
            y_val = np.asarray(target_val_labels, dtype=int)

            c_candidates = [0.01, 0.1, 1.0, 10.0]
            weight_candidates = [None, "balanced"]

            best_f1 = -1.0
            best_clf: Any = None
            best_c = 1.0
            best_weight: Any = None

            for c_val in c_candidates:
                for cw in weight_candidates:
                    try:
                        cand_clf = self._build_and_fit_classifier(c_val, cw, x_train, y_train)
                        val_preds = cand_clf.predict(x_val)
                        f1 = _compute_macro_f1(y_val, val_preds)
                    except Exception:
                        continue

                    if f1 > best_f1:
                        best_f1 = f1
                        best_clf = cand_clf
                        best_c = c_val
                        best_weight = cw

            if best_clf is not None:
                self.classifier_ = best_clf
                self.best_params_ = {
                    "model_type": self.model_type,
                    "C": best_c,
                    "class_weight": best_weight,
                    "val_macro_f1": float(best_f1),
                }
            else:
                # Fallback to default
                self.classifier_ = self._build_and_fit_classifier(1.0, None, x_train, y_train)
                self.best_params_ = {
                    "model_type": self.model_type,
                    "C": 1.0,
                    "class_weight": None,
                    "val_macro_f1": None,
                }
        else:
            # No validation set provided: fit default config C=1.0, class_weight=None
            self.classifier_ = self._build_and_fit_classifier(1.0, None, x_train, y_train)
            self.best_params_ = {
                "model_type": self.model_type,
                "C": 1.0,
                "class_weight": None,
                "val_macro_f1": None,
            }

        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Output 3-class probability distribution on simplex.

        Shape is (N, 3), rows sum strictly to 1.0, and columns correspond
        to [0: negative, 1: neutral, 2: positive].
        """
        if self.classifier_ is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")

        x = self.vectorizer.transform(texts)
        raw_probs = self.classifier_.predict_proba(x)

        # Ensure full 3-class alignment if any class was missing in training split
        n_samples = len(texts)
        probs = np.zeros((n_samples, 3), dtype=np.float64)

        classes = getattr(self.classifier_, "classes_", np.array([0, 1, 2]))
        for idx, cls in enumerate(classes):
            if 0 <= cls < 3:
                probs[:, cls] = raw_probs[:, idx]

        # Strict simplex normalization
        row_sums = probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        zero_rows = (row_sums.squeeze(axis=1) <= 1e-12)
        if np.any(zero_rows):
            probs[zero_rows] = 1.0 / 3.0
            row_sums[zero_rows] = 1.0

        probs = probs / row_sums
        return probs

    def predict(self, texts: list[str]) -> np.ndarray:
        """Output 3-class predicted integer labels [0, 1, 2]."""
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

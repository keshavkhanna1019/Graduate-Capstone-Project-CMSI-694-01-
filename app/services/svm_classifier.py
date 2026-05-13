"""
SVM face classifier trained on top of ArcFace embeddings.
Provides sharper per-user probability scores and unknown-face rejection.
"""
import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple

SVM_MODEL_PATH = "svm_classifier.pkl"


class SVMFaceClassifier:
    def __init__(self):
        self.clf = None
        self.label_map: Dict[int, str] = {}   # int index → user_id
        self.is_trained = False
        self.n_users = 0
        self.n_samples = 0

    def train(self, user_embeddings: Dict[str, List[List[float]]]) -> dict:
        """
        Train RBF-SVM on ArcFace embeddings.
        user_embeddings: {user_id: [embedding, ...]}  — at least 1 embedding per user.
        Requires >= 2 users.
        """
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        valid_users = {uid: embs for uid, embs in user_embeddings.items() if embs}
        if len(valid_users) < 2:
            raise ValueError(
                "Need at least 2 enrolled users with training photos to train the SVM. "
                "Enroll a second user first."
            )

        X, y = [], []
        self.label_map = {}
        for idx, uid in enumerate(sorted(valid_users)):
            self.label_map[idx] = uid
            for emb in valid_users[uid]:
                X.append(np.array(emb, dtype=np.float64))
                y.append(idx)

        X = np.array(X)
        y = np.array(y)

        # Pipeline: standardise → SVM with Platt scaling for probabilities
        self.clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
            )),
        ])
        self.clf.fit(X, y)

        self.is_trained = True
        self.n_users = len(valid_users)
        self.n_samples = len(X)

        return {
            "n_users": self.n_users,
            "n_samples": self.n_samples,
            "users": sorted(valid_users.keys()),
        }

    def predict(
        self,
        embedding: List[float],
        threshold: float = 0.45,
    ) -> Tuple[List[Tuple[str, float]], str, float, bool]:
        """
        Predict user from an ArcFace embedding.

        Returns:
          all_probs  — [(user_id, probability), ...] sorted descending
          best_user  — user_id with highest probability
          best_prob  — that probability
          is_match   — True if best_prob >= threshold
        """
        if not self.is_trained or self.clf is None:
            raise ValueError("SVM classifier is not trained yet. Call train-svm first.")

        x = np.array(embedding, dtype=np.float64).reshape(1, -1)
        proba = self.clf.predict_proba(x)[0]

        all_probs = sorted(
            [(self.label_map[i], float(p)) for i, p in enumerate(proba)],
            key=lambda t: t[1],
            reverse=True,
        )
        best_user, best_prob = all_probs[0]
        return all_probs, best_user, best_prob, best_prob >= threshold

    def save(self, path: str = SVM_MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump({
                "clf": self.clf,
                "label_map": self.label_map,
                "n_users": self.n_users,
                "n_samples": self.n_samples,
            }, f)

    def load(self, path: str = SVM_MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.clf = data["clf"]
            self.label_map = data["label_map"]
            self.n_users = data.get("n_users", len(self.label_map))
            self.n_samples = data.get("n_samples", 0)
            self.is_trained = True
            return True
        except Exception:
            return False


# Module-level singleton — loaded once on first access
_instance: Optional[SVMFaceClassifier] = None


def get_svm_classifier() -> SVMFaceClassifier:
    global _instance
    if _instance is None:
        _instance = SVMFaceClassifier()
        _instance.load(SVM_MODEL_PATH)
    return _instance


def reload_svm_classifier():
    """Force-reload from disk (call after training)."""
    global _instance
    _instance = SVMFaceClassifier()
    _instance.load(SVM_MODEL_PATH)
    return _instance

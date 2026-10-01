from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.text_extraction import clean_text

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "risk_model.joblib"
RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def load_model():
    return joblib.load(MODEL_PATH)


def classify_clauses(bundle, clauses, threshold=None):
    threshold = bundle["threshold"] if threshold is None else threshold
    vec, clf = bundle["vec"], bundle["clf"]
    names = np.array(vec.get_feature_names_out())
    X = vec.transform([clean_text(c) for c in clauses])
    proba = clf.predict_proba(X)
    results = []
    for i, p in enumerate(proba):
        idx = int(p.argmax())
        contrib = X[i].toarray()[0] * clf.coef_[idx]
        top = np.argsort(contrib)[::-1][:5]
        results.append({
            "clause": clauses[i],
            "risk": clf.classes_[idx],
            "confidence": round(float(p[idx]), 3),
            "needs_review": bool(p[idx] < threshold),
            "top_terms": ", ".join(names[j] for j in top if contrib[j] > 0),
        })
    return results


def similar_clauses(bundle, clause, n=3):
    x = bundle["vec"].transform([clean_text(clause)])
    sims = cosine_similarity(x, bundle["X_train"])[0]
    nn = np.argsort(sims)[::-1][:n]
    return [{
        "similarity": round(float(sims[i]), 3),
        "clause_type": bundle["train_label"][i],
        "risk": bundle["train_risk"][i],
        "text": bundle["train_text"][i],
    } for i in nn]


def summarize(results):
    risks = [r["risk"] for r in results]
    return {
        "document_risk": max(risks, key=lambda r: RISK_ORDER[r]),
        "clauses": len(results),
        "high": risks.count("High"),
        "medium": risks.count("Medium"),
        "low": risks.count("Low"),
        "review": sum(r["needs_review"] for r in results),
    }
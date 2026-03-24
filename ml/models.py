"""ml/models.py — All 4 ML models: train, evaluate, save, load."""
import os, sys, joblib
import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics        import (accuracy_score, roc_auc_score, f1_score,
                                    precision_score, recall_score, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


# ── Model Factories ───────────────────────────────────────────────────────────

def make_lr():
    return LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs",
        max_iter=1000, random_state=C.RANDOM_STATE)

def make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, criterion="gini",
        min_samples_split=5, n_jobs=-1, random_state=C.RANDOM_STATE)

def make_xgb():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, random_state=C.RANDOM_STATE)

def make_dnn():
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu",
        solver="adam", alpha=1e-4,
        learning_rate_init=1e-3, max_iter=300,
        random_state=C.RANDOM_STATE)


MODEL_FACTORIES = {
    "Logistic Regression": make_lr,
    "Random Forest":       make_rf,
    "GradientBoosting":    make_xgb,
    "Deep Neural Network": make_dnn,
}

MODEL_KEYS = {
    "Logistic Regression": "lr",
    "Random Forest":       "rf",
    "GradientBoosting":    "xgb",
    "Deep Neural Network": "dnn",
}


# ── Train / Evaluate ──────────────────────────────────────────────────────────

def train(model, X_tr, y_tr, sample_weight=None):
    kw = {}
    if sample_weight is not None:
        try:    kw["sample_weight"] = sample_weight
        except Exception: pass
    model.fit(X_tr, y_tr, **kw)
    return model


def evaluate(model, X_te, y_te) -> dict:
    yp  = model.predict(X_te)
    ypr = model.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_te, yp).ravel())
    return dict(
        accuracy =round(accuracy_score(y_te, yp), 4),
        auc_roc  =round(roc_auc_score(y_te, ypr), 4),
        f1       =round(f1_score(y_te, yp), 4),
        precision=round(precision_score(y_te, yp, zero_division=0), 4),
        recall   =round(recall_score(y_te, yp, zero_division=0), 4),
        tn=tn, fp=fp, fn=fn, tp=tp,
        confusion_matrix=[[tn, fp], [fn, tp]])


def proba(model, X):  return model.predict_proba(X)[:, 1]
def pred(model, X):   return model.predict(X)


# ── Save / Load ───────────────────────────────────────────────────────────────

def save(model, name):
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    path = os.path.join(C.MODELS_DIR, f"{MODEL_KEYS[name]}.pkl")
    joblib.dump(model, path)
    return path


def load(name):
    path = os.path.join(C.MODELS_DIR, f"{MODEL_KEYS.get(name, name)}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_all():
    out = {}
    for name in MODEL_FACTORIES:
        try:    out[name] = load(name)
        except FileNotFoundError: pass
    return out
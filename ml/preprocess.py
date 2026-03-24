"""ml/preprocess.py  —  Feature engineering, encoding, scaling, splitting."""
import os, sys, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from data.loader import get_df


def build_splits(df: pd.DataFrame):
    """
    Returns X_train, X_test, y_train, y_test,
            sex_train, sex_test, feature_names, scaler
    """
    drop_cols = [c for c in ["credit_risk","sex","age_young","personal_status"]
                 if c in df.columns]
    feat_df   = df.drop(columns=drop_cols)

    cat_use   = [c for c in C.CATEGORICAL_COLS if c in feat_df.columns]
    feat_df   = pd.get_dummies(feat_df, columns=cat_use, drop_first=False)
    feat_names = list(feat_df.columns)

    X   = feat_df.values.astype(np.float32)
    y   = df["credit_risk"].values
    sex = df["sex"].values

    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, sex, test_size=C.TEST_SIZE,
        random_state=C.RANDOM_STATE, stratify=y)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    return X_tr_sc, X_te_sc, y_tr, y_te, s_tr, s_te, feat_names, scaler


def save_artifacts(scaler, feat_names):
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    joblib.dump(scaler,     os.path.join(C.MODELS_DIR, "scaler.pkl"))
    joblib.dump(feat_names, os.path.join(C.MODELS_DIR, "feature_names.pkl"))


def load_scaler():
    return joblib.load(os.path.join(C.MODELS_DIR, "scaler.pkl"))


def load_feat_names():
    return joblib.load(os.path.join(C.MODELS_DIR, "feature_names.pkl"))


def preprocess_single(raw_dict: dict, scaler, feat_names) -> np.ndarray:
    """Transform one applicant dict into a scaled feature vector."""
    cat_use = [c for c in C.CATEGORICAL_COLS if c != "personal_status"]
    num_use = C.NUMERICAL_COLS
    row = {c: [raw_dict.get(c, "A14")] for c in cat_use}
    row.update({c: [float(raw_dict.get(c, 0))] for c in num_use})
    df_row = pd.DataFrame(row)
    df_enc = pd.get_dummies(df_row, columns=cat_use, drop_first=False)
    df_aln = df_enc.reindex(columns=feat_names, fill_value=0)
    return scaler.transform(df_aln.values.astype(np.float32))

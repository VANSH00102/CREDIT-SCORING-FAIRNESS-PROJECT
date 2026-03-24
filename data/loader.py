"""
data/loader.py  —  Download, load, and enrich the German Credit Dataset.
Protected attribute 'sex' derived from 'personal_status':
  A91 = male divorced/sep    → sex = 1
  A92 = female div/sep/mar   → sex = 0
  A93 = male single          → sex = 1
  A94 = male married/widowed → sex = 1
  A95 = female single        → sex = 0
Target: 1 (Good) → 0,  2 (Bad) → 1
"""
import os, sys, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def download():
    os.makedirs(C.DATA_DIR, exist_ok=True)
    if os.path.exists(C.DATASET_PATH):
        return C.DATASET_PATH
    print("[INFO] Downloading German Credit Dataset from UCI ...")
    try:
        urllib.request.urlretrieve(C.DATASET_URL, C.DATASET_PATH)
        print(f"[INFO] Saved → {C.DATASET_PATH}")
    except Exception as e:
        print(f"[WARN] Download failed ({e}). Using pre-generated dataset.")
        if not os.path.exists(C.DATASET_PATH):
            raise FileNotFoundError(
                "Dataset not found. Place german.data in the data/ folder.")
    return C.DATASET_PATH


def load_raw() -> pd.DataFrame:
    download()
    df = pd.read_csv(C.DATASET_PATH, sep=r"\s+", header=None, names=C.COLUMN_NAMES)
    df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})   # 0=Good, 1=Bad
    return df


def add_protected(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sex"] = df["personal_status"].apply(
        lambda x: 0 if x in ("A92", "A95") else 1)
    df["age_young"] = (df["age"] < 25).astype(int)
    return df


def get_df() -> pd.DataFrame:
    return add_protected(load_raw())


if __name__ == "__main__":
    df = get_df()
    print(df.shape, df["credit_risk"].value_counts().to_dict(),
          df["sex"].value_counts().to_dict())

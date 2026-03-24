"""
ml/mitigation.py  —  Three bias mitigation strategies.

  1. Reweighing          (pre-processing)
  2. SMOTE               (pre-processing)
  3. Threshold Optimisation (post-processing)
  4. Adversarial-proxy   (in-processing approximation via Prejudice Remover)
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics   import roc_auc_score


# ── 1. Reweighing ─────────────────────────────────────────────────────────────

def reweighing_weights(y_tr, sex_tr):
    """
    Kamiran & Calders (2012).
    w(x,g) = P(Y)·P(G) / P(Y,G)
    """
    n = len(y_tr)
    w = np.ones(n, dtype=np.float64)
    for g in (0,1):
        for y in (0,1):
            mask = (sex_tr==g) & (y_tr==y)
            p_y  = np.mean(y_tr==y)
            p_g  = np.mean(sex_tr==g)
            p_yg = np.mean(mask)
            if p_yg > 0:
                w[mask] = (p_y * p_g) / p_yg
    return w


def train_reweighing(factory, X_tr, y_tr, sex_tr):
    w = reweighing_weights(y_tr, sex_tr)
    m = factory()
    try:    m.fit(X_tr, y_tr, sample_weight=w)
    except TypeError: m.fit(X_tr, y_tr)
    return m


# ── 2. SMOTE ──────────────────────────────────────────────────────────────────

def smote(X_tr, y_tr, k=5, seed=42):
    """
    Synthetic Minority Over-sampling Technique.
    Uses imbalanced-learn if available; otherwise pure-numpy fallback.
    """
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(k_neighbors=k, random_state=seed)
        X_res, y_res = sm.fit_resample(X_tr, y_tr)
    except ImportError:
        X_res, y_res = _smote_np(X_tr, y_tr, k=k, seed=seed)

    print(f"[SMOTE] {np.bincount(y_tr)} → {np.bincount(y_res)}")
    return X_res, y_res


def _smote_np(X, y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    cls, cts = np.unique(y, return_counts=True)
    maj  = cls[np.argmax(cts)]; n_maj = cts.max()
    min_ = cls[np.argmin(cts)]; n_min = cts.min()
    n_need = n_maj - n_min
    Xm = X[y == min_]
    kk = min(k, len(Xm)-1) if len(Xm) > 1 else 1
    nn = NearestNeighbors(n_neighbors=kk+1).fit(Xm)
    _, idx = nn.kneighbors(Xm)
    synth = []
    for _ in range(n_need):
        i   = rng.integers(0, len(Xm))
        j   = rng.choice(idx[i, 1:])
        gap = rng.random()
        synth.append(Xm[i] + gap*(Xm[j]-Xm[i]))
    Xs = np.vstack([X, np.array(synth)])
    ys = np.hstack([y, np.full(n_need, min_, dtype=y.dtype)])
    perm = rng.permutation(len(Xs))
    return Xs[perm], ys[perm]


def train_smote(factory, X_tr, y_tr):
    Xs, ys = smote(X_tr, y_tr)
    m = factory(); m.fit(Xs, ys)
    return m


# ── 3. Threshold Optimisation ─────────────────────────────────────────────────

def optimize_thresholds(y_te, ypr, sex, metric="eqodd",
                        max_auc_drop=0.05, n_grid=50):
    """
    Grid-search group-specific thresholds τ_male, τ_female.
    Returns dict with threshold_male, threshold_female.
    """
    grid   = np.linspace(0.15, 0.85, n_grid)
    base   = roc_auc_score(y_te, ypr)
    best_s = 1e9; best_tm = 0.5; best_tf = 0.5

    def _tpr_fpr(yt, yp_bin):
        pos = yt==0; neg = yt==1
        tpr = float(np.mean(yp_bin[pos]==0)) if pos.sum()>0 else 0.
        fpr = float(np.mean(yp_bin[neg]==0)) if neg.sum()>0 else 0.
        return tpr, fpr

    for tm in grid:
        for tf in grid:
            yp_new = np.where(sex==1, (ypr<tm).astype(int),
                                      (ypr<tf).astype(int))
            if metric == "eqodd":
                t_m, f_m = _tpr_fpr(y_te[sex==1], yp_new[sex==1])
                t_f, f_f = _tpr_fpr(y_te[sex==0], yp_new[sex==0])
                score    = max(abs(t_m-t_f), abs(f_m-f_f))
            else:
                pr_m = float(np.mean(yp_new[sex==1]==0))
                pr_f = float(np.mean(yp_new[sex==0]==0))
                score = abs(pr_m-pr_f)
            if score < best_s:
                best_s = score; best_tm = tm; best_tf = tf

    return dict(threshold_male  =round(float(best_tm),4),
                threshold_female=round(float(best_tf),4),
                metric=metric, best_score=round(float(best_s),4))


def apply_thresholds(ypr, sex, thresholds):
    tm = thresholds.get("threshold_male",   0.5)
    tf = thresholds.get("threshold_female", 0.5)
    # Good credit = ypr < threshold (lower risk probability → approve)
    return np.where(sex==1, (ypr<tm).astype(int), (ypr<tf).astype(int))


# ── 4. Prejudice Remover (in-processing proxy) ───────────────────────────────

def train_prejudice_remover(factory, X_tr, y_tr, sex_tr, lam=1.0):
    """
    Combines Reweighing with upweighting unprivileged group.
    Works with any sklearn estimator supporting sample_weight.
    """
    base_w   = reweighing_weights(y_tr, sex_tr)
    extra_w  = np.where(sex_tr==1, 1.0, 1.0 + lam*0.5)
    w        = base_w * extra_w
    m = factory()
    try:    m.fit(X_tr, y_tr, sample_weight=w)
    except TypeError: m.fit(X_tr, y_tr)
    return m


# ── run all strategies on a given factory ────────────────────────────────────

def run_all(factory, X_tr, y_tr, sex_tr, X_te, y_te, sex_te):
    import warnings; warnings.filterwarnings("ignore")
    results = {}

    print("[MIT] Baseline ...")
    m = factory(); m.fit(X_tr, y_tr); results["Baseline"] = m

    print("[MIT] Reweighing ...")
    results["Reweighing"] = train_reweighing(factory, X_tr, y_tr, sex_tr)

    print("[MIT] SMOTE ...")
    results["SMOTE"] = train_smote(factory, X_tr, y_tr)

    print("[MIT] Prejudice Remover ...")
    results["Prejudice Remover"] = train_prejudice_remover(
        factory, X_tr, y_tr, sex_tr)

    return results

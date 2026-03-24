"""
ml/fairness.py  —  All 5 fairness metrics used in the paper.
Protected attribute: sex  (0 = female / unprivileged, 1 = male / privileged)
Target:             credit_risk  (0 = Good, 1 = Bad)
Positive outcome:  prediction = 0  (loan APPROVED / Good credit)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ── helpers ──────────────────────────────────────────────────────────────────

def _tpr(yt, yp):
    """TPR for 'approved' = prediction 0, good label = 0."""
    pos = yt == 0
    return float(np.mean(yp[pos] == 0)) if pos.sum() > 0 else 0.0

def _fpr(yt, yp):
    neg = yt == 1
    return float(np.mean(yp[neg] == 0)) if neg.sum() > 0 else 0.0

def _pr(yp):
    return float(np.mean(yp == 0))


# ── individual metrics ────────────────────────────────────────────────────────

def demographic_parity_difference(yp, sex):
    pr_m = _pr(yp[sex==1]);  pr_f = _pr(yp[sex==0])
    dpd  = pr_m - pr_f
    return dict(dpd=round(abs(dpd),4), dpd_signed=round(dpd,4),
                pr_male=round(pr_m,4), pr_female=round(pr_f,4))

def disparate_impact_ratio(yp, sex):
    pr_m = _pr(yp[sex==1]);  pr_f = _pr(yp[sex==0])
    ratio = pr_f / pr_m if pr_m > 0 else 0.0
    return dict(dir=round(ratio,4), compliant=bool(ratio >= 0.80),
                pr_male=round(pr_m,4), pr_female=round(pr_f,4))

def equal_opportunity_difference(yt, yp, sex):
    t_m = _tpr(yt[sex==1], yp[sex==1])
    t_f = _tpr(yt[sex==0], yp[sex==0])
    return dict(eod=round(abs(t_m-t_f),4), tpr_male=round(t_m,4),
                tpr_female=round(t_f,4))

def equalized_odds_difference(yt, yp, sex):
    t_m = _tpr(yt[sex==1], yp[sex==1]); t_f = _tpr(yt[sex==0], yp[sex==0])
    f_m = _fpr(yt[sex==1], yp[sex==1]); f_f = _fpr(yt[sex==0], yp[sex==0])
    dt  = abs(t_m-t_f);  df = abs(f_m-f_f)
    return dict(eqodd=round(max(dt,df),4),
                delta_tpr=round(dt,4), delta_fpr=round(df,4),
                tpr_male=round(t_m,4), tpr_female=round(t_f,4),
                fpr_male=round(f_m,4), fpr_female=round(f_f,4))

def group_auc(yt, ypr, sex):
    def _auc(yt_g, ypr_g):
        if len(np.unique(yt_g)) < 2: return float("nan")
        return float(roc_auc_score(yt_g, ypr_g))
    am = _auc(yt[sex==1], ypr[sex==1])
    af = _auc(yt[sex==0], ypr[sex==0])
    return dict(auc_male=round(am,4), auc_female=round(af,4),
                auc_gap=round(abs(am-af),4))


# ── combined ──────────────────────────────────────────────────────────────────

def all_metrics(yt, yp, ypr, sex, model_name="Model") -> dict:
    d = demographic_parity_difference(yp, sex)
    r = disparate_impact_ratio(yp, sex)
    e = equal_opportunity_difference(yt, yp, sex)
    q = equalized_odds_difference(yt, yp, sex)
    a = group_auc(yt, ypr, sex)
    return dict(model=model_name,
                n_male=int((sex==1).sum()), n_female=int((sex==0).sum()),
                dpd=d["dpd"], dpd_signed=d["dpd_signed"],
                pr_male=d["pr_male"], pr_female=d["pr_female"],
                dir=r["dir"], dir_compliant=r["compliant"],
                eod=e["eod"], tpr_male=e["tpr_male"], tpr_female=e["tpr_female"],
                eqodd=q["eqodd"], delta_tpr=q["delta_tpr"], delta_fpr=q["delta_fpr"],
                fpr_male=q["fpr_male"], fpr_female=q["fpr_female"],
                auc_male=a["auc_male"], auc_female=a["auc_female"],
                auc_gap=a["auc_gap"])


# ── Shapley-Lorenz (feature-level attribution) ────────────────────────────────

def shapley_lorenz(model, X_te, sex, feat_names, n_bg=100):
    """
    Tries SHAP KernelExplainer; falls back to permutation importance
    difference between groups.
    Returns {feature: sl_contribution}  (top-15, sorted descending)
    """
    try:
        import shap, warnings
        warnings.filterwarnings("ignore")
        bg   = X_te[:min(n_bg, len(X_te))]
        exp  = shap.KernelExplainer(model.predict_proba, bg, link="logit")
        sv   = exp.shap_values(X_te[:50], nsamples=100)
        sv   = sv[1] if isinstance(sv, list) else sv
        m_m  = np.abs(sv[sex[:50]==1]).mean(0)
        m_f  = np.abs(sv[sex[:50]==0]).mean(0)
        out  = {fn: round(float(abs(m-f)),6)
                for fn,(m,f) in zip(feat_names,zip(m_m,m_f))}
    except Exception:
        out = _perm_attribution(model, X_te, sex, feat_names)
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True)[:15])


def _perm_attribution(model, X_te, sex, feat_names):
    base = model.predict_proba(X_te)[:,1]
    out  = {}
    rng  = np.random.default_rng(42)
    for i, fn in enumerate(feat_names):
        Xp = X_te.copy(); Xp[:,i] = rng.permutation(Xp[:,i])
        pm = model.predict_proba(Xp)[:,1]
        diff_priv   = np.abs(base[sex==1]-pm[sex==1]).mean()
        diff_unpriv = np.abs(base[sex==0]-pm[sex==0]).mean()
        out[fn]     = round(float(abs(diff_priv-diff_unpriv)),6)
    return out


# ── summary table ─────────────────────────────────────────────────────────────

def summary_table(results: list) -> pd.DataFrame:
    rows = [dict(Model=r["model"],
                 **{"DPD↓":r["dpd"],"DIR":r["dir"],
                    "Compliant":"✓" if r["dir_compliant"] else "✗",
                    "EOD↓":r["eod"],"EqODD↓":r["eqodd"],
                    "AUC Gap↓":r["auc_gap"],
                    "TPR Male":r["tpr_male"],"TPR Female":r["tpr_female"]})
            for r in results]
    return pd.DataFrame(rows)

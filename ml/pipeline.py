"""
ml/pipeline.py  —  Full training + fairness + mitigation pipeline.
Run standalone:  python ml/pipeline.py
"""
import os, sys, json, time, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from data.loader       import get_df
from ml.preprocess     import build_splits, save_artifacts
from ml.models         import (MODEL_FACTORIES, train, evaluate,
                               save, proba, pred)
from ml.fairness       import all_metrics, shapley_lorenz, summary_table
from ml.mitigation     import (run_all, optimize_thresholds,
                               apply_thresholds)

import joblib

STATUS_F  = os.path.join(C.MODELS_DIR, "status.json")
RESULTS_F = os.path.join(C.MODELS_DIR, "results.json")


# ── status helpers ────────────────────────────────────────────────────────────

def set_status(d):
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    with open(STATUS_F, "w") as f: json.dump(d, f, indent=2)

def get_status():
    if not os.path.exists(STATUS_F):
        return {"status": "not_started"}
    with open(STATUS_F) as f: return json.load(f)

def save_results(d):
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    with open(RESULTS_F, "w") as f: json.dump(d, f, indent=2, default=str)

def load_results():
    if not os.path.exists(RESULTS_F): return {}
    with open(RESULTS_F) as f: return json.load(f)


# ── plot helpers ──────────────────────────────────────────────────────────────

COLORS = {"Logistic Regression":"#1F4E9A","Random Forest":"#1A7A4A",
          "GradientBoosting":"#D68910","Deep Neural Network":"#C0392B"}

def _pd(): os.makedirs(C.PLOTS_DIR, exist_ok=True)

def plot_roc(roc_data):
    _pd(); fig, ax = plt.subplots(figsize=(8,6))
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random (0.50)")
    for name,(fpr,tpr,a) in roc_data.items():
        ax.plot(fpr,tpr,lw=2.2,color=COLORS.get(name,"steelblue"),
                label=f"{name} (AUC={a:.3f})")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve — All Models",fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    p=os.path.join(C.PLOTS_DIR,"roc.png")
    plt.tight_layout(); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    return "roc.png"

def plot_fairness_bar(baseline, mitigated):
    _pd(); fig, axes = plt.subplots(1,3,figsize=(15,5))
    keys  = [("dpd","DPD ↓"),("eqodd","EqODD ↓"),("dir","DIR ↑")]
    labels= [m["model"] for m in baseline]
    x     = np.arange(len(labels)); w=0.35
    for ax,(key,ylabel) in zip(axes,keys):
        b_vals=[m[key] for m in baseline]; m_vals=[m[key] for m in mitigated]
        ax.bar(x-w/2,b_vals,w,label="Baseline",color="#C0392B",alpha=0.85)
        ax.bar(x+w/2,m_vals,w,label="Mitigated",color="#1A7A4A",alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right",fontsize=8)
        ax.set_title(ylabel,fontweight="bold"); ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3)
        if key=="dir": ax.axhline(0.80,color="orange",lw=1.5,ls="--",label="EEOC 0.80")
    fig.suptitle("Fairness Metrics: Baseline vs Mitigated",fontweight="bold")
    p=os.path.join(C.PLOTS_DIR,"fairness_bar.png")
    plt.tight_layout(); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    return "fairness_bar.png"

def plot_tradeoff(base_fm):
    _pd(); fig,ax = plt.subplots(figsize=(8,6))
    for m in base_fm:
        c=COLORS.get(m["model"],"steelblue")
        ax.scatter(m["auc_roc"],m["dpd"],s=130,color=c,zorder=3,label=m["model"])
        ax.annotate(m["model"].split()[0],(m["auc_roc"],m["dpd"]),
                    textcoords="offset points",xytext=(6,4),fontsize=8)
    ax.axvline(0.80,color="#1F4E9A",lw=1.5,ls="--",alpha=0.7)
    ax.axhline(0.10,color="#D68910",lw=1.5,ls="--",alpha=0.7)
    ax.set_xlabel("AUC-ROC"); ax.set_ylabel("DPD (lower = fairer)")
    ax.set_title("Accuracy vs Fairness Trade-off",fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    p=os.path.join(C.PLOTS_DIR,"tradeoff.png")
    plt.tight_layout(); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    return "tradeoff.png"

def plot_approval(baseline, mitigated):
    _pd(); fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(mlist,title) in zip(axes,[
        (baseline,"Baseline Approval Rates"),
        (mitigated,"Mitigated Approval Rates")]):
        labels=[m["model"] for m in mlist]; x=np.arange(len(labels)); w=0.35
        male_r  =[m["pr_male"]   for m in mlist]
        female_r=[m["pr_female"] for m in mlist]
        ax.bar(x-w/2,[1-v for v in male_r],  w,label="Male",  color="#1F4E9A",alpha=0.85)
        ax.bar(x+w/2,[1-v for v in female_r],w,label="Female",color="#C0392B",alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right",fontsize=8)
        ax.set_ylabel("Approval Rate"); ax.set_title(title,fontweight="bold")
        ax.set_ylim(0,1); ax.legend(); ax.grid(axis="y",alpha=0.3)
    p=os.path.join(C.PLOTS_DIR,"approval.png")
    plt.tight_layout(); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    return "approval.png"

def plot_shapley(sl_data, model_name):
    if not sl_data: return None
    _pd(); items=list(sl_data.items())[:12]
    feats,vals=zip(*items)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(range(len(feats)),vals,color="#1F4E9A",alpha=0.85)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats,fontsize=9)
    ax.set_xlabel("Shapley-Lorenz Contribution")
    ax.set_title(f"Feature Bias Attribution — {model_name}",fontweight="bold")
    ax.grid(axis="x",alpha=0.3)
    p=os.path.join(C.PLOTS_DIR,"shapley.png")
    plt.tight_layout(); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    return "shapley.png"


# ── main pipeline ─────────────────────────────────────────────────────────────

def run():
    set_status({"status":"running","step":"Loading data","progress":5})
    print("\n"+"="*55)
    print("  CREDIT FAIRNESS — TRAINING PIPELINE")
    print("="*55)

    # 1. Data
    set_status({"status":"running","step":"Preprocessing data","progress":10})
    df = get_df()
    X_tr,X_te,y_tr,y_te,s_tr,s_te,fnames,scaler = build_splits(df)
    save_artifacts(scaler, fnames)
    print(f"\n[DATA] train={X_tr.shape}  test={X_te.shape}  features={len(fnames)}")
    print(f"       male_test={int((s_te==1).sum())}  female_test={int((s_te==0).sum())}")

    # 2. Train all models
    set_status({"status":"running","step":"Training models","progress":20})
    print("\n[MODELS] Training 4 models ...")
    trained, roc_data, perf = {}, {}, []

    for i,(name,factory) in enumerate(MODEL_FACTORIES.items()):
        t0=time.time(); print(f"  → {name} ...",end="",flush=True)
        m  = train(factory(), X_tr, y_tr)
        ev = evaluate(m, X_te, y_te)
        ev["model"] = name; ev["train_time"] = round(time.time()-t0,2)
        perf.append(ev); trained[name]=m; save(m, name)
        fpr_,tpr_,_=roc_curve(y_te, proba(m,X_te))
        roc_data[name]=(fpr_.tolist(),tpr_.tolist(),ev["auc_roc"])
        print(f"  AUC={ev['auc_roc']}  Acc={ev['accuracy']}  [{ev['train_time']}s]")
        set_status({"status":"running","step":f"Trained {name}",
                    "progress":20+i*10})

    # 3. Baseline fairness
    set_status({"status":"running","step":"Computing fairness metrics","progress":60})
    print("\n[FAIRNESS] Baseline metrics ...")
    base_fm = []
    for name,m in trained.items():
        yp_=pred(m,X_te); ypr_=proba(m,X_te)
        fm =all_metrics(y_te,yp_,ypr_,s_te,name)
        ev =next(e for e in perf if e["model"]==name)
        fm.update({"auc_roc":ev["auc_roc"],"accuracy":ev["accuracy"],"f1":ev["f1"]})
        base_fm.append(fm)
        tag="✓" if fm["dir_compliant"] else "✗"
        print(f"  {name:25s}  DPD={fm['dpd']}  DIR={fm['dir']} {tag}  EqODD={fm['eqodd']}")

    # 4. Mitigation on GradientBoosting
    set_status({"status":"running","step":"Running mitigation strategies","progress":70})
    print("\n[MIT] Mitigation on GradientBoosting ...")
    mit_models = run_all(MODEL_FACTORIES["GradientBoosting"],
                         X_tr,y_tr,s_tr, X_te,y_te,s_te)
    mit_fm = []
    for strat,m in mit_models.items():
        yp_=pred(m,X_te); ypr_=proba(m,X_te)
        fm=all_metrics(y_te,yp_,ypr_,s_te,strat)
        ev=evaluate(m,X_te,y_te)
        fm.update({"auc_roc":ev["auc_roc"],"accuracy":ev["accuracy"]})
        mit_fm.append(fm)
        tag="✓" if fm["dir_compliant"] else "✗"
        print(f"  {strat:25s}  DPD={fm['dpd']}  DIR={fm['dir']} {tag}  AUC={fm['auc_roc']}")

    # Threshold optimisation (post-processing)
    set_status({"status":"running","step":"Threshold optimisation","progress":78})
    print("\n[THRESH] Optimising thresholds ...")
    xgb_m   = trained["GradientBoosting"]
    ypr_xgb = proba(xgb_m, X_te)
    thresholds = optimize_thresholds(y_te, ypr_xgb, s_te)
    yp_thresh  = apply_thresholds(ypr_xgb, s_te, thresholds)
    fm_thresh  = all_metrics(y_te, yp_thresh, ypr_xgb, s_te, "Threshold Opt.")
    fm_thresh.update({"auc_roc":round(roc_auc_score(y_te,ypr_xgb),4),
                      "accuracy":round(float(np.mean(yp_thresh==y_te)),4)})
    mit_fm.append(fm_thresh)
    joblib.dump(thresholds, os.path.join(C.MODELS_DIR,"thresholds.pkl"))
    tag="✓" if fm_thresh["dir_compliant"] else "✗"
    print(f"  {'Threshold Opt.':25s}  DPD={fm_thresh['dpd']}  "
          f"DIR={fm_thresh['dir']} {tag}  AUC={fm_thresh['auc_roc']}")

    # 5. Shapley-Lorenz
    set_status({"status":"running","step":"Shapley-Lorenz attribution","progress":83})
    print("\n[SL] Shapley-Lorenz attribution ...")
    sl={}
    try:
        sl=shapley_lorenz(xgb_m, X_te, s_te, fnames)
        print(f"  Top-3: {list(sl.items())[:3]}")
    except Exception as ex:
        print(f"  [WARN] {ex}")

    # 6. Build mitigated comparison (threshold opt for all models)
    mit_all = []
    for name,m in trained.items():
        ypr_=proba(m,X_te)
        t=optimize_thresholds(y_te,ypr_,s_te)
        yp_t=apply_thresholds(ypr_,s_te,t)
        fm=all_metrics(y_te,yp_t,ypr_,s_te,name)
        fm["auc_roc"]=round(roc_auc_score(y_te,ypr_),4)
        mit_all.append(fm)

    # 7. Plots
    set_status({"status":"running","step":"Generating plots","progress":88})
    print("\n[PLOTS] ...")
    plots={}
    plots["roc"]         = plot_roc(roc_data)
    plots["fairness_bar"]= plot_fairness_bar(base_fm, mit_all)
    plots["tradeoff"]    = plot_tradeoff(base_fm)
    plots["approval"]    = plot_approval(base_fm, mit_all)
    plots["shapley"]     = plot_shapley(sl, "GradientBoosting")
    print(f"  Saved: {list(plots.values())}")

    # 8. Persist results
    results = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        dataset="German Credit Dataset (UCI)",
        n_train=int(X_tr.shape[0]), n_test=int(X_te.shape[0]),
        n_features=len(fnames),
        n_male_test=int((s_te==1).sum()), n_female_test=int((s_te==0).sum()),
        feature_names=fnames,
        performance=perf,
        baseline_fairness=base_fm,
        mitigation_results=mit_fm,
        mitigation_all=mit_all,
        shapley_lorenz=sl,
        thresholds=thresholds,
        plots=plots,
    )
    save_results(results)
    set_status({"status":"done","step":"Complete","progress":100})

    # 9. Print summary
    print("\n"+"-"*55)
    print(f"{'Model':25s} {'AUC':>7} {'Acc':>7} {'DPD':>7} {'DIR':>7}  Compliant")
    print("-"*55)
    for fm in base_fm:
        print(f"{fm['model']:25s} {fm['auc_roc']:>7.4f} {fm['accuracy']:>7.4f} "
              f"{fm['dpd']:>7.4f} {fm['dir']:>7.4f}  "
              f"{'✓' if fm['dir_compliant'] else '✗'}")
    print("\n  PIPELINE COMPLETE ✓")
    return results


if __name__ == "__main__":
    run()

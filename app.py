"""
app.py  —  Flask Web Application
Routes:
  GET  /                  → Dashboard
  GET  /train             → Training page
  POST /api/train         → Trigger pipeline (background thread)
  GET  /api/train/status  → Poll training progress
  GET  /predict           → Prediction form
  POST /api/predict       → Predict single applicant
  GET  /fairness          → Fairness dashboard
  GET  /api/fairness      → Fairness JSON
  GET  /compare           → Model comparison
  GET  /api/compare       → Comparison JSON
  GET  /api/results       → Full results JSON
"""
import os, sys, json, threading, traceback
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from ml.pipeline   import run, get_status, load_results, set_status
from ml.models     import load_all, proba, pred
from ml.preprocess import load_scaler, load_feat_names, preprocess_single
from ml.fairness   import all_metrics
from ml.mitigation import apply_thresholds

app = Flask(__name__)
app.config["SECRET_KEY"] = C.SECRET_KEY

for d in (C.DATA_DIR, C.MODELS_DIR, C.PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

_THREAD = None


# ── page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    status  = get_status()
    results = load_results() if status.get("status") == "done" else {}
    stats   = {}
    if results:
        best = max(results.get("baseline_fairness",[]), key=lambda x:x.get("auc_roc",0), default={})
        stats = dict(n_train=results.get("n_train","-"),
                     n_test =results.get("n_test","-"),
                     n_feat =results.get("n_features","-"),
                     best_model=best.get("model","-"),
                     best_auc  =best.get("auc_roc","-"))
    return render_template("index.html", status=status, stats=stats)

@app.route("/train")
def train_page():
    return render_template("train.html", status=get_status())

@app.route("/predict")
def predict_page():
    ready = get_status().get("status") == "done"
    return render_template("predict.html", models_ready=ready)

@app.route("/fairness")
def fairness_page():
    return render_template("fairness.html", status=get_status(),
                           results=load_results())

@app.route("/compare")
def compare_page():
    return render_template("compare.html", status=get_status(),
                           results=load_results())


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def api_train():
    global _THREAD
    if get_status().get("status") == "running":
        return jsonify({"error":"Training already running"}), 409
    def _run():
        try:  run()
        except Exception as e:
            set_status({"status":"error","message":str(e),"progress":0})
            traceback.print_exc()
    _THREAD = threading.Thread(target=_run, daemon=True)
    _THREAD.start()
    return jsonify({"message":"started"})

@app.route("/api/train/status")
def api_status():
    return jsonify(get_status())

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if get_status().get("status") != "done":
        return jsonify({"error":"Models not trained yet"}), 400
    data       = request.get_json(force=True)
    model_name = data.pop("model_name", "GradientBoosting")
    sex_str    = data.pop("sex", "male")
    sex_val    = 0 if sex_str == "female" else 1

    try:
        models    = load_all()
        scaler    = load_scaler()
        fnames    = load_feat_names()
        results   = load_results()

        if model_name not in models:
            return jsonify({"error":f"Model '{model_name}' not found"}), 404

        m = models[model_name]
        X = preprocess_single(data, scaler, fnames)
        ypr = float(proba(m, X)[0])          # probability of BAD credit

        # Group-specific threshold (if available and XGBoost)
        threshold = 0.5
        thresh_path = os.path.join(C.MODELS_DIR, "thresholds.pkl")
        if os.path.exists(thresh_path) and model_name == "GradientBoosting":
            th = joblib.load(thresh_path)
            threshold = th["threshold_male"] if sex_val==1 else th["threshold_female"]

        y_pred   = int(ypr >= threshold)        # 1=Bad, 0=Good
        approved = y_pred == 0

        # Risk tier
        if   ypr < 0.20: tier, tc = "Very Low Risk",  "success"
        elif ypr < 0.40: tier, tc = "Low Risk",        "success"
        elif ypr < 0.60: tier, tc = "Moderate Risk",   "warning"
        elif ypr < 0.80: tier, tc = "High Risk",       "danger"
        else:            tier, tc = "Very High Risk",  "danger"

        fnote = ""
        if sex_val==0 and threshold != 0.5:
            fnote = (f"Fairness-adjusted threshold ({threshold:.2f}) applied "
                     f"to reduce demographic disparity for female applicants.")

        return jsonify(dict(
            prediction=y_pred, approved=approved,
            label="Good Credit Risk" if approved else "Bad Credit Risk",
            prob_bad=round(ypr,4), prob_good=round(1-ypr,4),
            risk_tier=tier, tier_color=tc,
            model_used=model_name, threshold=round(threshold,4),
            sex="Female" if sex_val==0 else "Male",
            fairness_note=fnote,
        ))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}), 500

@app.route("/api/fairness")
def api_fairness():
    r = load_results()
    if not r: return jsonify({"error":"No results yet"}), 400
    return jsonify(dict(baseline_fairness  =r.get("baseline_fairness",[]),
                        mitigation_results =r.get("mitigation_results",[]),
                        shapley_lorenz     =r.get("shapley_lorenz",{}),
                        thresholds         =r.get("thresholds",{}),
                        plots              =r.get("plots",{})))

@app.route("/api/compare")
def api_compare():
    r = load_results()
    if not r: return jsonify({"error":"No results yet"}), 400
    return jsonify(dict(performance      =r.get("performance",[]),
                        baseline_fairness=r.get("baseline_fairness",[]),
                        n_train=r.get("n_train"), n_test=r.get("n_test"),
                        n_features=r.get("n_features")))

@app.route("/api/results")
def api_results():
    return jsonify(load_results())

@app.route("/static/images/<filename>")
def serve_img(filename):
    return send_from_directory(C.PLOTS_DIR, filename)

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html",
                           status=get_status(), stats={}), 404

if __name__ == "__main__":
    print(f"\n{'='*50}\n  CreditFairAI  →  http://localhost:{C.PORT}\n{'='*50}\n")
    app.run(debug=C.DEBUG, port=C.PORT, use_reloader=False)

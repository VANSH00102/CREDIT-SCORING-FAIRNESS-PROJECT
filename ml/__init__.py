# ml/__init__.py  —  Credit Fairness ML Package
# Exposes all public functions for import convenience

from .preprocess import (
    build_splits,
    save_artifacts,
    load_scaler,
    load_feat_names,
    preprocess_single,
)

from .models import (
    MODEL_FACTORIES,
    MODEL_KEYS,
    train,
    evaluate,
    save,
    load,
    load_all,
    proba,
    pred,
)

from .fairness import (
    all_metrics,
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    group_auc,
    shapley_lorenz,
    summary_table,
)

from .mitigation import (
    reweighing_weights,
    train_reweighing,
    smote,
    train_smote,
    optimize_thresholds,
    apply_thresholds,
    train_prejudice_remover,
    run_all,
)

from .pipeline import (
    run,
    get_status,
    set_status,
    load_results,
    save_results,
)

__all__ = [
    # preprocess
    "build_splits", "save_artifacts", "load_scaler",
    "load_feat_names", "preprocess_single",
    # models
    "MODEL_FACTORIES", "MODEL_KEYS", "train", "evaluate",
    "save", "load", "load_all", "proba", "pred",
    # fairness
    "all_metrics", "demographic_parity_difference",
    "disparate_impact_ratio", "equal_opportunity_difference",
    "equalized_odds_difference", "group_auc",
    "shapley_lorenz", "summary_table",
    # mitigation
    "reweighing_weights", "train_reweighing", "smote",
    "train_smote", "optimize_thresholds", "apply_thresholds",
    "train_prejudice_remover", "run_all",
    # pipeline
    "run", "get_status", "set_status", "load_results", "save_results",
]

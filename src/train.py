import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
try:
    from sklearn.model_selection import StratifiedGroupKFold
    CV_CLASS = StratifiedGroupKFold
except ImportError:
    from sklearn.model_selection import GroupKFold
    CV_CLASS = GroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from cogstate import make_dataset, save_bundle

ROOT = os.path.join(os.path.dirname(__file__), "..")
X, y, groups, meta = make_dataset(os.path.join(ROOT, "EEG_dataset"))
np.savez(os.path.join(ROOT, "test", "features.npz"), X=X, y=y, groups=groups)
models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "linear_svc_calibrated": make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=0), method="sigmoid", cv=3)),
    "histgb": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, random_state=0),
}

def score(y_true, p):
    return {"bACC": float(balanced_accuracy_score(y_true, p >= 0.5)), "AUC": float(roc_auc_score(y_true, p))}

def grouped_scores(y_true, p, meta_rows, key):
    vals = {}
    for i, row in enumerate(meta_rows):
        group_key = (row["subject"], row["trial_id"]) if key == "trial_id" else row[key]
        vals.setdefault(group_key, [[], []]); vals[group_key][0].append(y_true[i]); vals[group_key][1].append(p[i])
    bacc = [balanced_accuracy_score(v[0], np.asarray(v[1]) >= 0.5) for v in vals.values()]
    auc = [roc_auc_score(v[0], v[1]) for v in vals.values() if len(set(v[0])) == 2]
    return {"n": len(vals), "bACC_mean": float(np.mean(bacc)), "AUC_mean": float(np.mean(auc)) if auc else None}

res = {"n_samples": int(len(y)), "n_subjects": int(len(set(groups))), "n_features": int(X.shape[1]), "cv": "StratifiedGroupKFold" if CV_CLASS.__name__ == "StratifiedGroupKFold" else "GroupKFold", "models": {}}
for name, model in models.items():
    fold_rows = []
    all_p = np.full(len(y), np.nan)
    if CV_CLASS.__name__ == "StratifiedGroupKFold":
        cv = CV_CLASS(n_splits=5, shuffle=True, random_state=0)
        splits = cv.split(X, y, groups)
    else:
        splits = CV_CLASS(n_splits=5).split(X, y, groups)
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        model.fit(X[train_idx], y[train_idx])
        p = model.predict_proba(X[valid_idx])[:, 1]
        all_p[valid_idx] = p
        fold_rows.append({"fold": fold, "n": int(len(valid_idx)), "subjects": sorted(set(groups[valid_idx].tolist())), **score(y[valid_idx], p)})
        print(f"{name} fold {fold}: {fold_rows[-1]}", flush=True)
    result = {**score(y, all_p), "folds": fold_rows, "subject": grouped_scores(y, all_p, meta, "subject"), "trial": grouped_scores(y, all_p, meta, "trial_id")}
    res["models"][name] = result
    model.fit(X, y)
    save_bundle(model, os.path.join(ROOT, "test", f"{name}.joblib"))
    print(name, result, flush=True)
with open(os.path.join(ROOT, "test", "results.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

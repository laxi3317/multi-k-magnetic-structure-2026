import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

def get_oof_proba(clf, X, y, cv, cfg):
    oof = np.zeros(len(y))
    smote_tgt = cfg["smote_target"]
    seed = cfg["seed"]
    min_k = cfg["smote_k_min"]
    X_arr = np.array(X)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_arr, y)):
        X_tr, X_val = X_arr[tr_idx].copy(), X_arr[val_idx].copy()
        y_tr, y_val = y[tr_idx].copy(), y[val_idx].copy()
        n_pos = int((y_tr == 1).sum())
        if n_pos < smote_tgt:
            k_nei = min(min_k, max(1, n_pos - 1))
            try:
                sm = SMOTE(sampling_strategy={1:smote_tgt}, k_neighbors=k_nei, random_state=seed)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception as e:
                pass
        clf.fit(X_tr, y_tr)
        oof[val_idx] = clf.predict_proba(X_val)[:, 1]
    return oof

def find_best_threshold(proba, y_true, cfg):
    st = cfg["threshold_start"]
    ed = cfg["threshold_end"]
    step = cfg["threshold_step"]
    rec_floor = cfg["recall_floor"]
    prec_floor = cfg["precision_floor"]
    mf1_floor = cfg["macro_floor_base"]
    mf1_fallback = cfg["macro_floor_fallback"]
    ths = np.arange(st, ed, step)
    res = []
    for th in ths:
        pred = (proba >= th).astype(int)
        rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
        prec = precision_score(y_true, pred, pos_label=1, zero_division=0)
        prec_z = precision_score(y_true, pred, pos_label=0, zero_division=0)
        f1_arr = f1_score(y_true, pred, average=None, zero_division=0)
        mf1 = f1_score(y_true, pred, average="macro", zero_division=0)
        f1_nz = f1_arr[1] if len(f1_arr) > 1 else 0.0
        f1_z = f1_arr[0] if len(f1_arr) > 0 else 0.0
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        res.append({
            "threshold": round(th,3),
            "Precision": round(prec,4),
            "Precision_z": round(prec_z,4),
            "Recall": round(rec,4),
            "F1_nonzero": round(f1_nz,4),
            "F1_zero": round(f1_z,4),
            "Macro_F1": round(mf1,4),
            "TP": tp, "FP": fp, "FN": fn
        })
    df_r = pd.DataFrame(res)
    valid = df_r[(df_r["Recall"] >= rec_floor) & (df_r["Precision"] >= prec_floor) & (df_r["Macro_F1"] >= mf1_floor)]
    if len(valid) == 0:
        valid = df_r[(df_r["Recall"] >= rec_floor) & (df_r["Macro_F1"] >= mf1_fallback)]
    if len(valid) == 0:
        best = df_r.loc[df_r["Macro_F1"].idxmax()]
    else:
        best = valid.loc[valid["Macro_F1"].idxmax()]
    return best["threshold"], df_r, best

def retrain_and_predict(clf, X_tr, y_tr, X_te, cfg):
    smote_tgt = cfg["smote_target"]
    seed = cfg["seed"]
    min_k = cfg["smote_k_min"]
    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr)
    n_pos = int((y_tr == 1).sum())
    if n_pos < smote_tgt:
        k_nei = min(min_k, max(1, n_pos - 1))
        try:
            sm = SMOTE(sampling_strategy={1:smote_tgt}, k_neighbors=k_nei, random_state=seed)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        except Exception as e:
            pass
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(np.array(X_te))[:,1]

def full_eval(name, proba, y_true, threshold, label=""):
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    mf1 = f1_score(y_true, pred, average="macro", zero_division=0)
    f1s = f1_score(y_true, pred, average=None, zero_division=0)
    rcs = recall_score(y_true, pred, average=None, zero_division=0)
    prec_pos = precision_score(y_true, pred, pos_label=1, zero_division=0)
    prec_neg = precision_score(y_true, pred, pos_label=0, zero_division=0)
    cm = confusion_matrix(y_true, pred)
    f1_z = f1s[0] if len(f1s) > 0 else 0.0
    rec_z = rcs[0] if len(rcs) > 0 else 0.0
    f1_nz = f1s[1] if len(f1s) > 1 else 0.0
    rec_nz = rcs[1] if len(rcs) > 1 else 0.0
    return {
        "model": name,
        "threshold": threshold,
        "AUC": round(auc,4),
        "Macro_F1": round(mf1,4),
        "F1_zero": round(f1_z,4),
        "Recall_zero": round(rec_z,4),
        "Prec_zero": round(prec_neg,4),
        "F1_nonzero": round(f1_nz,4),
        "Recall_nonzero": round(rec_nz,4),
        "Prec_nonzero": round(prec_pos,4),
    }
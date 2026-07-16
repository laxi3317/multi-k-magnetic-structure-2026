import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from config import *

def get_oof_prob(clf, X, y, cv):
    oof_prob = np.zeros(len(y))
    X_arr = np.array(X)
    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_arr, y)):
        X_tr, X_val = X_arr[tr_idx].copy(), X_arr[val_idx].copy()
        y_tr, y_val = y[tr_idx].copy(), y[val_idx].copy()
        pos_num = int((y_tr == 1).sum())
        if pos_num < SMOTE_TARGET:
            k_nei = min(5, max(1, pos_num - 1))
            try:
                sm = SMOTE(sampling_strategy={1: SMOTE_TARGET}, k_neighbors=k_nei, random_state=SEED)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception as e:
                print(f"{fold_idx+1}SMOTE：{e}，jump")
        clf.fit(X_tr, y_tr)
        oof_prob[val_idx] = clf.predict_proba(X_val)[:, 1]
        print(f"{fold_idx+1}/{cv.get_n_splits()}finish | single:{(y_tr==0).sum()} multi:{(y_tr==1).sum()}")
    return oof_prob

def search_opt_threshold(proba, y_true):
    threshold_list = np.arange(TH_START, TH_END, TH_STEP)
    res_list = []
    for th in threshold_list:
        pred = (proba >= th).astype(int)
        rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
        prec = precision_score(y_true, pred, pos_label=1, zero_division=0)
        f1_all = f1_score(y_true, pred, average=None, zero_division=0)
        macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        f1_multi = f1_all[1] if len(f1_all) > 1 else 0.0
        f1_single = f1_all[0] if len(f1_all) > 0 else 0.0
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        res_list.append({
            "threshold": round(th, 3),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_multi": round(f1_multi, 4),
            "F1_single": round(f1_single, 4),
            "Macro_F1": round(macro_f1, 4),
            "TP": tp, "FP": fp, "FN": fn
        })
    df_res = pd.DataFrame(res_list)
    valid_df = df_res[(df_res["Recall"] >= RECALL_FLOOR) & (df_res["Precision"] >= PRECISION_FLOOR) & (df_res["Macro_F1"] >= 0.70)]
    if len(valid_df) == 0:
        valid_df = df_res[(df_res["Recall"] >= RECALL_FLOOR) & (df_res["Macro_F1"] >= 0.68)]
    if len(valid_df) == 0:
        best_row = df_res.loc[df_res["Macro_F1"].idxmax()]
    else:
        best_row = valid_df.loc[valid_df["Macro_F1"].idxmax()]
    return best_row["threshold"], df_res, best_row

def retrain_predict_test(clf, X_train, y_train, X_test):
    X_tr = np.array(X_train)
    y_tr = np.array(y_train)
    pos_num = int((y_tr == 1).sum())
    if pos_num < SMOTE_TARGET:
        k_nei = min(5, max(1, pos_num - 1))
        try:
            sm = SMOTE(sampling_strategy={1: SMOTE_TARGET}, k_neighbors=k_nei, random_state=SEED)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        except Exception as e:
            print(f"SMOTE：{e}")
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(np.array(X_test))[:, 1]

def evaluate_model(name, proba, y_true, threshold, data_tag=""):
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    f1_arr = f1_score(y_true, pred, average=None, zero_division=0)
    rec_arr = recall_score(y_true, pred, average=None, zero_division=0)
    prec_multi = precision_score(y_true, pred, pos_label=1, zero_division=0)

    f1_s = f1_arr[0] if len(f1_arr) > 0 else 0.0
    rec_s = rec_arr[0] if len(rec_arr) > 0 else 0.0
    f1_m = f1_arr[1] if len(f1_arr) > 1 else 0.0
    rec_m = rec_arr[1] if len(rec_arr) > 1 else 0.0

    print(f"\n{'='*55}")
    print(f"model：{name}  best_th={threshold:.3f}  {data_tag}")
    print(f"{'='*55}")
    print(classification_report(y_true, pred, target_names=["single_k", "multi_k"], digits=4))
    print(f"AUC-ROC  = {auc:.4f}")
    print(f"Macro-F1 = {macro_f1:.4f}  {'✅' if macro_f1 >= 0.75 else '❌'}")
    print(f"multi_k F1 = {f1_m:.4f}  {'✅' if f1_m >= 0.58 else '❌'}")
    print(f"multi_k Recall = {rec_m:.4f}  {'✅' if rec_m >= 0.70 else '❌'}")
    print(f"multi_k Precision = {prec_multi:.4f}")
    cm = confusion_matrix(y_true, pred)
    print(f"{'':15} {'Psingle_k':>14} {'Pmulti_k':>14}")
    print(f"{'Tsingle_k':15} {cm[0,0]:>14} {cm[0,1]:>14}")
    print(f"{'Tmulti_k':15} {cm[1,0]:>14} {cm[1,1]:>14}")

    return {
        "model": name,
        "threshold": round(threshold, 4),
        "AUC": round(auc, 4),
        "Macro_F1": round(macro_f1, 4),
        "F1_multi": round(f1_m, 4),
        "Recall_multi": round(rec_m, 4),
        "Prec_multi": round(prec_multi, 4),
        "F1_single": round(f1_s, 4),
        "Recall_single": round(rec_s, 4)
    }
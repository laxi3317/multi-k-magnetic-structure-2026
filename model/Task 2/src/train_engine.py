import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
import lightgbm as lgb
from xgboost import XGBClassifier
from config import *
from model_utils import get_oof_prob, search_opt_threshold, retrain_predict_test, evaluate_model

def init_base_models(pos_weight):
    pw_boost = min(pos_weight * 1.5, 25.0)
    lgb = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=pw_boost)
    cw_rf = {0:1, 1: max(1, int(round(pw_boost)))}
    rf = RandomForestClassifier(**RF_ET_PARAMS, class_weight=cw_rf)
    xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=pw_boost)
    cw_et = {0:1, 1: max(1, int(round(pw_boost)))}
    et = ExtraTreesClassifier(**RF_ET_PARAMS, class_weight=cw_et)
    return (lgb, rf, xgb, et), pw_boost

def search_ensemble_weight(oof_lgb, oof_rf, oof_xgb, oof_et, y_train):
    best_info = None
    best_mf1 = 0.0
    for w_lgb in np.arange(0.1, 0.6, 0.1):
        for w_rf in np.arange(0.05, 0.5, 0.1):
            for w_et in np.arange(0.1, 0.6, 0.1):
                w_xgb = round(1 - w_lgb - w_rf - w_et, 2)
                if w_xgb < 0.05 or w_xgb > 0.6:
                    continue
                combine_prob = w_lgb*oof_lgb + w_rf*oof_rf + w_xgb*oof_xgb + w_et*oof_et
                th, _, _ = search_opt_threshold(combine_prob, y_train)
                pred = (combine_prob >= th).astype(int)
                rec = recall_score(y_train, pred, pos_label=1, zero_division=0)
                if rec < RECALL_FLOOR:
                    continue
                prec = precision_score(y_train, pred, pos_label=1, zero_division=0)
                if prec < 0.3:
                    continue
                mf1 = f1_score(y_train, pred, average="macro", zero_division=0)
                if mf1 < 0.7:
                    continue
                f1_arr = f1_score(y_train, pred, average=None, zero_division=0)
                f1_m = f1_arr[1] if len(f1_arr) > 1 else 0
                if mf1 > best_mf1:
                    best_mf1 = mf1
                    best_info = {
                        "w_lgb": w_lgb, "w_rf": w_rf, "w_xgb": w_xgb, "w_et": w_et,
                        "threshold": th, "Precision": prec, "Recall": rec,
                        "F1_multi": f1_m, "Macro_F1": mf1
                    }
    return best_info

def train_stacking_meta(X_stack_train, y_train, X_stack_test, cv, pw_boost):
    best_c = None
    best_mf1 = 0.0
    best_oof = None
    best_test_prob = None
    for c in LR_C_LIST:
        oof_p = np.zeros(len(y_train))
        test_p = np.zeros(len(X_stack_test))
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=c, class_weight={0:1, 1: max(1, int(round(pw_boost)))},
                solver="lbfgs", max_iter=LR_MAX_ITER, random_state=SEED
            ))
        ])
        for tr_idx, val_idx in cv.split(X_stack_train, y_train):
            pipe.fit(X_stack_train[tr_idx], y_train[tr_idx])
            oof_p[val_idx] = pipe.predict_proba(X_stack_train[val_idx])[:, 1]
            test_p += pipe.predict_proba(X_stack_test)[:, 1]
        test_p /= cv.get_n_splits()
        th_c, _, row_c = search_opt_threshold(oof_p, y_train)
        pred = (oof_p >= th_c).astype(int)
        mf1 = f1_score(y_train, pred, average="macro", zero_division=0)
        rec = recall_score(y_train, pred, pos_label=1, zero_division=0)
        if mf1 > best_mf1 and rec >= RECALL_FLOOR:
            best_mf1 = mf1
            best_c = c
            best_oof = oof_p.copy()
            best_test_prob = test_p.copy()
    if best_oof is None:
        for c in LR_C_LIST:
            oof_p = np.zeros(len(y_train))
            test_p = np.zeros(len(X_stack_test))
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    C=c, class_weight={0:1, 1: max(1, int(round(pw_boost)))},
                    solver="lbfgs", max_iter=LR_MAX_ITER, random_state=SEED
                ))
            ])
            for tr_idx, val_idx in cv.split(X_stack_train, y_train):
                pipe.fit(X_stack_train[tr_idx], y_train[tr_idx])
                oof_p[val_idx] = pipe.predict_proba(X_stack_train[val_idx])[:, 1]
                test_p += pipe.predict_proba(X_stack_test)[:, 1]
            test_p /= cv.get_n_splits()
            pred = (oof_p >= 0.3).astype(int)
            mf1 = f1_score(y_train, pred, average="macro", zero_division=0)
            if mf1 > best_mf1:
                best_mf1 = mf1
                best_c = c
                best_oof = oof_p.copy()
                best_test_prob = test_p.copy()
    return best_oof, best_test_prob, best_c

def draw_roc_pr_curve(y_true, pred_dict, th_dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_pr, ax_roc = axes[0], axes[1]
    for name, (prob, _) in pred_dict.items():
        p, r, _ = precision_recall_curve(y_true, prob)
        auc_pr = float(np.trapz(p[::-1], r[::-1]))
        ax_pr.plot(r, p, label=f"{name} AUC={auc_pr:.3f}", color=COLORS[name], lw=2)
        th = th_dict[name]
        pred_op = (prob >= th).astype(int)
        rec_op = recall_score(y_true, pred_op, pos_label=1, zero_division=0)
        pre_op = precision_score(y_true, pred_op, pos_label=1, zero_division=0)
        ax_pr.scatter(rec_op, pre_op, color=COLORS[name], s=100, edgecolors="black")
    ax_pr.axhline(0.5, ls="--", c="gray", alpha=0.5)
    ax_pr.set_xlabel("Recall (multi_k)")
    ax_pr.set_ylabel("Precision (multi_k)")
    ax_pr.set_title("Precision-Recall Curve (Test Set)")
    ax_pr.legend()
    ax_pr.grid(alpha=0.3)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    for name, (prob, _) in pred_dict.items():
        auc_roc = roc_auc_score(y_true, prob)
        fpr, tpr, _ = roc_curve(y_true, prob)
        ax_roc.plot(fpr, tpr, label=f"{name} AUC={auc_roc:.3f}", color=COLORS[name], lw=2)
        th = th_dict[name]
        pred_op = (prob >= th).astype(int)
        tpr_op = recall_score(y_true, pred_op, pos_label=1, zero_division=0)
        fpr_op = 1 - recall_score(y_true, pred_op, pos_label=0, zero_division=0)
        ax_roc.scatter(fpr_op, tpr_op, color=COLORS[name], s=100, edgecolors="black")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve (Test Set)")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    plt.suptitle("Multi-k Binary Classification Hold-out Evaluation", fontsize=14, weight="bold")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "classification_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"cure save：{save_path}")

def save_all_outputs(df_raw, idx_train, idx_test,
                     test_stack, test_ens, test_et, th_stack,
                     oof_lgb, oof_rf, oof_xgb, oof_et, oof_stack,
                     feat_cols, imputer, vt, scaler, cols_kept, cols_sel,
                     X_tr_sel, X_te_sel, y_train, y_test,
                     th_lgb, th_ens, pos_weight, pw_boost, best_ensemble, best_c):
    test_df = pd.DataFrame({
        "filename": df_raw.iloc[idx_test]["filename"].values,
        "label": df_raw.iloc[idx_test]["label"].values,
        "label_binary": df_raw.iloc[idx_test]["label_binary"].values,
        "proba_stack": test_stack,
        "proba_ensemble": test_ens,
        "proba_et": test_et,
        "pred_binary": (test_stack >= th_stack).astype(int),
        "split": "test"
    })
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    train_df = pd.DataFrame({
        "filename": df_raw.iloc[idx_train]["filename"].values,
        "label": df_raw.iloc[idx_train]["label"].values,
        "label_binary": df_raw.iloc[idx_train]["label_binary"].values,
        "oof_lgb": oof_lgb,
        "oof_rf": oof_rf,
        "oof_xgb": oof_xgb,
        "oof_et": oof_et,
        "oof_stack": oof_stack,
        "split": "train_oof"
    })
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_oof.csv"), index=False)
    test_label_bin = (df_raw.iloc[idx_test]["label_binary"] == "multi_k").astype(int)
    err_df = test_df[(test_stack >= th_stack).astype(int) != test_label_bin].copy()
    err_df.to_csv(os.path.join(OUTPUT_DIR, "error_samples.csv"), index=False)
    pd.Series(feat_cols).to_csv(os.path.join(OUTPUT_DIR, "used_features.csv"), index=False, header=["feature"])
    joblib.dump(lgb_clf, os.path.join(MODEL_SAVE_DIR, "lgb_model.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_SAVE_DIR, "knn_imputer.pkl"))
    joblib.dump(vt, os.path.join(MODEL_SAVE_DIR, "variance_filter.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "robust_scaler.pkl"))
    with open(os.path.join(MODEL_SAVE_DIR, "vt_kept_cols.json"), "w", encoding="utf-8") as f:
        json.dump(cols_kept, f)
    with open(os.path.join(MODEL_SAVE_DIR, "final_selected_cols.json"), "w", encoding="utf-8") as f:
        json.dump(cols_sel, f)
    np.save(os.path.join(MODEL_SAVE_DIR, "X_train_sel.npy"), X_tr_sel)
    np.save(os.path.join(MODEL_SAVE_DIR, "X_test_sel.npy"), X_te_sel)
    np.save(os.path.join(MODEL_SAVE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(MODEL_SAVE_DIR, "y_test.npy"), y_test)
    cfg_save = {
        "th_lgb": float(th_lgb),
        "th_ensemble": float(th_ens),
        "pos_weight": float(pos_weight),
        "pw_boost": float(pw_boost),
        "best_ensemble_weight": best_ensemble if best_ensemble else {},
        "stack_best_C": best_c
    }
    with open(os.path.join(MODEL_SAVE_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_save, f, indent=2)
    test_names = df_raw.iloc[idx_test]["filename"].tolist()
    with open(os.path.join(MODEL_SAVE_DIR, "test_filename_list.json"), "w", encoding="utf-8") as f:
        json.dump(test_names, f)
    print(f"save{OUTPUT_DIR}")
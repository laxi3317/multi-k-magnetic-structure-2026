import argparse
import yaml
import numpy as np
import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from src.train import (
    load_and_filter_data, preprocess_pipeline,
    train_base_models, weight_grid_search, retrain_and_predict,
    find_best_threshold, full_eval, train_stacking_meta, plot_roc_pr_curve
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/train_cfg.yaml")
    args = parser.parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    np.random.seed(cfg["seed"])
    cv = StratifiedKFold(n_splits=cfg["cv_splits"], shuffle=True, random_state=cfg["seed"])
    df, feat_cols, X_tr_raw, X_te_raw, y_train, y_test, idx_tr, idx_te, pos_weight, fill_cnt, leak_hit, removed_exact, removed_substr = load_and_filter_data(cfg)
    X_tr_sel, X_te_sel, cols_kept, cols_sel, imp, vt, scaler, imp_scores, mask = preprocess_pipeline(X_tr_raw, X_te_raw, feat_cols, cfg, pos_weight)
    (lgb_clf, rf_clf, xgb_clf, et_clf), (oof_lgb, oof_rf, oof_xgb, oof_et, oof_ens), pw_boost = train_base_models(X_tr_sel, y_train, cv, pos_weight, cfg)
    th_lgb, _, _ = find_best_threshold(oof_lgb, y_train, cfg)
    th_rf, _, _ = find_best_threshold(oof_rf, y_train, cfg)
    th_xgb, _, _ = find_best_threshold(oof_xgb, y_train, cfg)
    th_et, _, _ = find_best_threshold(oof_et, y_train, cfg)
    th_ens, df_scan, _ = find_best_threshold(oof_ens, y_train, cfg)
    best_combo = weight_grid_search(oof_lgb, oof_rf, oof_xgb, oof_et, y_train, cfg)
    if best_combo:
        oof_weighted = best_combo["w_lgb"]*oof_lgb + best_combo["w_rf"]*oof_rf + best_combo["w_xgb"]*oof_xgb + best_combo["w_et"]*oof_et
        th_weighted = best_combo["threshold"]
    else:
        oof_weighted = oof_ens
        th_weighted = th_ens
    test_lgb = retrain_and_predict(lgb_clf, X_tr_sel, y_train, X_te_sel, cfg)
    test_rf = retrain_and_predict(rf_clf, X_tr_sel, y_train, X_te_sel, cfg)
    test_xgb = retrain_and_predict(xgb_clf, X_tr_sel, y_train, X_te_sel, cfg)
    test_et = retrain_and_predict(et_clf, X_tr_sel, y_train, X_te_sel, cfg)
    if best_combo:
        test_final = best_combo["w_lgb"]*test_lgb + best_combo["w_rf"]*test_rf + best_combo["w_xgb"]*test_xgb + best_combo["w_et"]*test_et
        th_final = best_combo["threshold"]
    else:
        test_final = 0.3*test_lgb + 0.2*test_rf + 0.2*test_xgb + 0.3*test_et
        th_final = th_ens
    oof_mat = np.column_stack([oof_lgb, oof_rf, oof_xgb, oof_et])
    X_stack_train = np.column_stack([
        oof_lgb, oof_rf, oof_xgb, oof_et,
        oof_lgb*oof_rf, oof_lgb*oof_xgb, oof_lgb*oof_et,
        oof_rf*oof_xgb, oof_rf*oof_et, oof_xgb*oof_et,
        np.max(oof_mat, axis=1), np.min(oof_mat, axis=1),
        np.std(oof_mat, axis=1), np.mean(oof_mat, axis=1),
    ])
    test_mat = np.column_stack([test_lgb, test_rf, test_xgb, test_et])
    X_stack_test = np.column_stack([
        test_lgb, test_rf, test_xgb, test_et,
        test_lgb*test_rf, test_lgb*test_xgb, test_lgb*test_et,
        test_rf*test_xgb, test_rf*test_et, test_xgb*test_et,
        np.max(test_mat, axis=1), np.min(test_mat, axis=1),
        np.std(test_mat, axis=1), np.mean(test_mat, axis=1),
    ])
    oof_stack, test_stack, best_C = train_stacking_meta(X_stack_train, y_train, X_stack_test, cv, pw_boost, cfg)
    th_stack, _, _ = find_best_threshold(oof_stack, y_train, cfg)
    oof_res = [
        full_eval("LGB   [Train OOF]", oof_lgb, y_train, th_lgb),
        full_eval("RF    [Train OOF]", oof_rf, y_train, th_rf),
        full_eval("XGB   [Train OOF]", oof_xgb, y_train, th_xgb),
        full_eval("ET    [Train OOF]", oof_et, y_train, th_et),
        full_eval("em  [Train OOF]", oof_ens, y_train, th_ens),
        full_eval("Stacking [Train OOF]", oof_stack, y_train, th_stack)
    ]
    test_res = [
        full_eval("LGB   [Test]", test_lgb, y_test, th_lgb, "★Hold-out"),
        full_eval("RF    [Test]", test_rf, y_test, th_rf, "★Hold-out"),
        full_eval("XGB   [Test]", test_xgb, y_test, th_xgb, "★Hold-out"),
        full_eval("ET    [Test]", test_et, y_test, th_et, "★Hold-out"),
        full_eval("em  [Test]", test_final, y_test, th_final, "★Hold-out"),
        full_eval("Stacking [Test]", test_stack, y_test, th_stack, "★Hold-out")
    ]
    df_test_summary = pd.DataFrame(test_res)
    print(df_test_summary[["model", "threshold", "AUC", "Macro_F1", "F1_zero", "Recall_zero", "Prec_zero", "F1_nonzero", "Recall_nonzero", "Prec_nonzero"]].to_string(index=False))
    best_idx = df_test_summary["Macro_F1"].idxmax()
    best_row = df_test_summary.loc[best_idx]
    stack_gap = oof_res[-1]["Macro_F1"] - test_res[-1]["Macro_F1"]
    pred_dict = {
        "LightGBM": (test_lgb, th_lgb),
        "RF": (test_rf, th_rf),
        "XGBoost": (test_xgb, th_xgb),
        "ET": (test_et, th_et),
        "Stacking": (test_stack, th_stack)
    }
    th_map = {
        "LightGBM": th_lgb,
        "RF": th_rf,
        "XGBoost": th_xgb,
        "ET": th_et,
        "Stacking": th_stack
    }
    plot_roc_pr_curve(y_test, pred_dict, th_map, cfg)
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(exist_ok=True)
    df_test_out = pd.DataFrame({
        "filename": df.iloc[idx_te]["filename"].values,
        "label": df.iloc[idx_te]["label"].values,
        "label_binary": df.iloc[idx_te]["label_binary"].values,
        "proba_stack": test_stack,
        "proba_ens": test_final,
        "proba_et": test_et,
        "pred_binary": (test_stack >= th_stack).astype(int),
        "split": "test",
    })
    df_test_out.to_csv("propagation_eval_test_predictions.csv", index=False)
    df_train_out = pd.DataFrame({
        "filename": df.iloc[idx_tr]["filename"].values,
        "label": df.iloc[idx_tr]["label"].values,
        "label_binary": df.iloc[idx_tr]["label_binary"].values,
        "oof_proba_lgb": oof_lgb,
        "oof_proba_rf": oof_rf,
        "oof_proba_xgb": oof_xgb,
        "oof_proba_et": oof_et,
        "oof_proba_stack": oof_stack,
        "split": "train_oof",
    })
    df_train_out.to_csv("propagation_eval_train_oof.csv", index=False)
    pred_test_final = (test_stack >= th_stack).astype(int)
    y_test_bin = (df.iloc[idx_te]["label_binary"] == "nonzero_k").astype(int)
    df_err = df_test_out[pred_test_final != y_test_bin].copy()
    df_err.to_csv("propagation_eval_test_errors.csv", index=False)
    pd.Series(feat_cols).to_csv("propagation_used_features.csv", index=False, header=["feature_name"])
    joblib.dump(lgb_clf, save_dir / "lgb_clf.pkl")
    joblib.dump(rf_clf, save_dir / "rf_clf.pkl")
    joblib.dump(xgb_clf, save_dir / "xgb_clf.pkl")
    joblib.dump(et_clf, save_dir / "et_clf.pkl")
    joblib.dump(imp, save_dir / "imputer.pkl")
    joblib.dump(vt, save_dir / "variance_threshold.pkl")
    joblib.dump(scaler, save_dir / "scaler.pkl")
    with open(save_dir / "cols_kept.json", "w", encoding="utf-8") as f:
        json.dump(cols_kept, f)
    with open(save_dir / "cols_sel.json", "w", encoding="utf-8") as f:
        json.dump(cols_sel, f)
    np.save(save_dir / "X_tr_sel.npy", X_tr_sel)
    np.save(save_dir / "X_te_sel.npy", X_te_sel)
    np.save(save_dir / "y_train.npy", y_train)
    np.save(save_dir / "y_test.npy", y_test)
    config_dump = {
        "task": "propagation_binary",
        "labels": {"0": "zero_k", "1": "nonzero_k"},
        "th_lgb": float(th_lgb),
        "th_final": float(th_final),
        "th_stack": float(th_stack),
        "pos_weight": float(pos_weight),
        "pw_boost": float(pw_boost),
        "best_combo": best_combo if best_combo else {},
        "best_c_stack": best_C
    }
    with open(save_dir / "prop_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

if __name__ == "__main__":
    main()
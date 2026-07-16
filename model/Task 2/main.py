import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from config import CV_SPLITS, SEED
from data_pipe import load_and_split_data, preprocess_pipeline
from train_engine import init_base_models, search_ensemble_weight, train_stacking_meta, draw_roc_pr_curve, save_all_outputs
from model_utils import search_opt_threshold, evaluate_model, retrain_predict_test

def main():
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)

    df_raw, feat_cols, X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test, pos_weight = load_and_split_data()

    X_tr_sel, X_te_sel, cols_kept, cols_sel, imputer, vt, scaler, imp_scores, sel_mask = preprocess_pipeline(X_train_raw, X_test_raw, feat_cols, pos_weight)

    (lgb_clf, rf_clf, xgb_clf, et_clf), pw_boost = init_base_models(pos_weight)

    oof_lgb = get_oof_prob(lgb_clf, X_tr_sel, y_train, cv)
    oof_rf = get_oof_prob(rf_clf, X_tr_sel, y_train, cv)
    oof_xgb = get_oof_prob(xgb_clf, X_tr_sel, y_train, cv)
    oof_et = get_oof_prob(et_clf, X_tr_sel, y_train, cv)
    oof_default_ens = 0.3*oof_lgb + 0.2*oof_rf + 0.2*oof_xgb + 0.3*oof_et

    th_lgb, _, _ = search_opt_threshold(oof_lgb, y_train)
    th_rf, _, _ = search_opt_threshold(oof_rf, y_train)
    th_xgb, _, _ = search_opt_threshold(oof_xgb, y_train)
    th_et, _, _ = search_opt_threshold(oof_et, y_train)
    th_default_ens, df_scan, _ = search_opt_threshold(oof_default_ens, y_train)

    best_ensemble = search_ensemble_weight(oof_lgb, oof_rf, oof_xgb, oof_et, y_train)
    if best_ensemble:
        oof_opt_ens = best_ensemble["w_lgb"]*oof_lgb + best_ensemble["w_rf"]*oof_rf + best_ensemble["w_xgb"]*oof_xgb + best_ensemble["w_et"]*oof_et
        th_opt_ens = best_ensemble["threshold"]
    else:
        oof_opt_ens = oof_default_ens
        th_opt_ens = th_default_ens

    test_lgb = retrain_predict_test(lgb_clf, X_tr_sel, y_train, X_te_sel)
    test_rf = retrain_predict_test(rf_clf, X_tr_sel, y_train, X_te_sel)
    test_xgb = retrain_predict_test(xgb_clf, X_tr_sel, y_train, X_te_sel)
    test_et = retrain_predict_test(et_clf, X_tr_sel, y_train, X_te_sel)
    if best_ensemble:
        test_opt_ens = best_ensemble["w_lgb"]*test_lgb + best_ensemble["w_rf"]*test_rf + best_ensemble["w_xgb"]*test_xgb + best_ensemble["w_et"]*test_et
        th_opt_ens_test = best_ensemble["threshold"]
    else:
        test_opt_ens = 0.3*test_lgb + 0.2*test_rf + 0.2*test_xgb + 0.3*test_et
        th_opt_ens_test = th_default_ens

    oof_mat = np.column_stack([oof_lgb, oof_rf, oof_xgb, oof_et])
    X_stack_train = np.column_stack([
        oof_lgb, oof_rf, oof_xgb, oof_et,
        oof_lgb*oof_rf, oof_lgb*oof_xgb, oof_lgb*oof_et,
        oof_rf*oof_xgb, oof_rf*oof_et, oof_xgb*oof_et,
        np.max(oof_mat, axis=1), np.min(oof_mat, axis=1), np.std(oof_mat, axis=1), np.mean(oof_mat, axis=1)
    ])
    test_mat = np.column_stack([test_lgb, test_rf, test_xgb, test_et])
    X_stack_test = np.column_stack([
        test_lgb, test_rf, test_xgb, test_et,
        test_lgb*test_rf, test_lgb*test_xgb, test_lgb*test_et,
        test_rf*test_xgb, test_rf*test_et, test_xgb*test_et,
        np.max(test_mat, axis=1), np.min(test_mat, axis=1), np.std(test_mat, axis=1), np.mean(test_mat, axis=1)
    ])

    oof_stack, test_stack, best_C = train_stacking_meta(X_stack_train, y_train, X_stack_test, cv, pw_boost)
    th_stack, _, _ = search_opt_threshold(oof_stack, y_train)
    oof_eval_res = [
        evaluate_model("LightGBM [OOF]", oof_lgb, y_train, th_lgb),
        evaluate_model("RF [OOF]", oof_rf, y_train, th_rf),
        evaluate_model("XGBoost [OOF]", oof_xgb, y_train, th_xgb),
        evaluate_model("ET [OOF]", oof_et, y_train, th_et),
        evaluate_model("Weight Ensemble [OOF]", oof_opt_ens, y_train, th_opt_ens),
        evaluate_model("Stacking [OOF]", oof_stack, y_train, th_stack),
    ]
    test_eval_res = [
        evaluate_model("LightGBM [Test]", test_lgb, y_test, th_lgb, "Hold-out"),
        evaluate_model("RF [Test]", test_rf, y_test, th_rf, "Hold-out"),
        evaluate_model("XGBoost [Test]", test_xgb, y_test, th_xgb, "Hold-out"),
        evaluate_model("ET [Test]", test_et, y_test, th_et, "Hold-out"),
        evaluate_model("Weight Ensemble [Test]", test_opt_ens, y_test, th_opt_ens_test, "Hold-out"),
        evaluate_model("Stacking [Test]", test_stack, y_test, th_stack, "Hold-out"),
    ]

    model_names = ["LightGBM", "RF", "XGBoost", "ET", "Weight Ensemble", "Stacking"]
    for idx, name in enumerate(model_names):
        oof_mf1 = oof_eval_res[idx]["Macro_F1"]
        test_mf1 = test_eval_res[idx]["Macro_F1"]
        gap = oof_mf1 - test_mf1
        if abs(gap) < 0.05:
            status = "✅"
        elif gap > 0.05:
            status = "⚠"
        else:
            status = "⚠"
        print(f"{name:<16} OOF_MF1:{oof_mf1:.4f} Test_MF1:{test_mf1:.4f} Gap:{gap:+.4f} {status}")

    plot_dict = {
        "LightGBM": (test_lgb, th_lgb),
        "RF": (test_rf, th_rf),
        "XGBoost": (test_xgb, th_xgb),
        "ET": (test_et, th_et),
        "Stacking": (test_stack, th_stack)
    }
    th_map = {k: v[1] for k, v in plot_dict.items()}
    draw_roc_pr_curve(y_test, plot_dict, th_map)
    save_all_outputs(
        df_raw, idx_train, idx_test,
        test_stack, test_opt_ens, test_et, th_stack,
        oof_lgb, oof_rf, oof_xgb, oof_et, oof_stack,
        feat_cols, imputer, vt, scaler, cols_kept, cols_sel,
        X_tr_sel, X_te_sel, y_train, y_test,
        th_lgb, th_opt_ens_test, pos_weight, pw_boost, best_ensemble, best_C
    )
    test_df_summary = pd.DataFrame(test_eval_res)
    print(test_df_summary[["model", "threshold", "AUC", "Macro_F1", "F1_multi", "Recall_multi"]].to_string(index=False))
    best_idx = test_df_summary["Macro_F1"].idxmax()
    best_row = test_df_summary.loc[best_idx]
    stack_gap = oof_eval_res[-1]["Macro_F1"] - test_eval_res[-1]["Macro_F1"]
    print(f"\nbest model：{best_row['model']}")
    print(f"Macro-F1 = {best_row['Macro_F1']:.4f}")
    print(f"multi_k F1 = {best_row['F1_multi']:.4f}")
    print(f"multi_k Recall = {best_row['Recall_multi']:.4f}")
    print(f"Stacking gap：{stack_gap:+.4f}")

if __name__ == "__main__":
    main()
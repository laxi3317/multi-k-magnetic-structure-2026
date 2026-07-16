import warnings
warnings.filterwarnings("ignore")
from config import *
from data_pipeline import load_split_data
from preprocessor import preprocess_all
from cascade_stage1 import train_stage1
from cascade_stage2 import train_stage2
from evaluate_viz import grid_search_best, draw_visual
from model_saver import save_all

def main():

    df_raw, feat_cols, X_train_raw, X_test_raw, y_tr_tri, y_te_tri, idx_train, idx_test, spw_s1 = load_split_data()

    X_tr_sel, X_te_sel, X_tr_sel_s2, X_te_sel_s2, imp, vt, scaler, et_sel, et_s2_feat, cols_kept, cols_sel, cols_sel_j = preprocess_all(X_train_raw, X_test_raw, feat_cols, spw_s1)

    s1_models, oof_s1_dict, test_prob_s1 = train_stage1(X_tr_sel, y_tr_bin, X_te_sel, spw_s1)

    s2_models, oof_s2_dict, contamination_rate, X_s2_os, y_s2_os, X_s2_noos, y_s2_noos = train_stage2(X_tr_sel_s2, y_tr_tri, y_tr_bin, oof_s1_dict, test_prob_s1, spw_s1)

    best_grid, best_conservative, y_pred_fin = grid_search_best(test_prob_s1, X_te_sel_s2, s2_models, y_te_tri)

    draw_visual(y_te_tri, y_pred_fin, best_grid, s2_models, test_prob_s1, X_te_sel_s2)

    save_all(
        s1_models, s2_models, imp, vt, scaler, et_sel, et_s2_feat,
        feat_cols, cols_sel, cols_sel_j,
        X_tr_sel, X_te_sel, X_tr_sel_s2, X_te_sel_s2,
        y_tr_tri, y_te_tri, best_grid, best_conservative, contamination_rate
    )


if __name__ == "__main__":
    main()
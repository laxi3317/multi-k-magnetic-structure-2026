import warnings
warnings.filterwarnings("ignore")

from data_pipeline import load_split_data
from preprocessor import preprocess_all
from cascade_stage1 import train_stage1
from cascade_stage2 import train_stage2
from evaluate_viz import grid_search_best
from model_saver import save_all

def main():
    df, feat_cols, Xtr_raw, Xte_raw, ytr_tri, yte_tri, ytr_bin, yte_bin, idxtr, idzte, spw1 = load_split_data()
    Xtr_sel, Xte_sel, Xtr_s2, Xte_s2, imp, vt, scaler, et1, et2, cols_kept, cols_sel, cols_sel_j = preprocess_all(Xtr_raw, Xte_raw, feat_cols, spw1)
    s1_models, oof1_dict, test_p1 = train_stage1(Xtr_sel, ytr_bin, Xte_sel, spw1)
    s2_models, cr, Xs2_os, ys2_os, Xs2_noos, ys2_noos = train_stage2(Xtr_s2, ytr_tri, ytr_bin, oof1_dict, test_p1, spw1)
    best_grid = grid_search_best(test_p1, Xte_s2, s2_models, yte_tri)
    save_all(s1_models, s2_models, imp, vt, scaler, et1, et2,
             feat_cols, cols_sel, cols_sel_j,
             Xtr_sel, Xte_sel, Xtr_s2, Xte_s2,
             ytr_tri, yte_tri, best_grid, None, cr)
    print("\n🎉 finish！")

if __name__ == "__main__":
    main()
import joblib
import json
import numpy as np
from config import SAVE_DIR

def save_all(s1_models, s2_models, imp, vt, scaler, et_sel, et_s2,
             feat_cols, cols_sel, cols_sel_j,
             X_tr_sel, X_te_sel, X_tr_sel_s2, X_te_sel_s2,
             y_tr_tri, y_te_tri, best_grid, best_conservative, cr):

    joblib.dump(s1_models["lgb"], SAVE_DIR/"s1_lgb.pkl")
    joblib.dump(s1_models["rf"],  SAVE_DIR/"s1_rf.pkl")
    joblib.dump(s1_models["xgb"], SAVE_DIR/"s1_xgb.pkl")
    joblib.dump(s1_models["et"],  SAVE_DIR/"s1_et.pkl")

    joblib.dump(s2_models["lgb"], SAVE_DIR/"s2_lgb.pkl")
    joblib.dump(s2_models["xgb"], SAVE_DIR/"s2_xgb.pkl")
    joblib.dump(s2_models["brf"], SAVE_DIR/"s2_brf.pkl")
    joblib.dump(s2_models["svm"], SAVE_DIR/"s2_svm.pkl")
    joblib.dump(s2_models["gb"],  SAVE_DIR/"s2_gb.pkl")

    joblib.dump(imp, SAVE_DIR/"knn_imputer.pkl")
    joblib.dump(vt, SAVE_DIR/"variance_threshold.pkl")
    joblib.dump(scaler, SAVE_DIR/"robust_scaler.pkl")
    joblib.dump(et_sel, SAVE_DIR/"et_selector.pkl")
    joblib.dump(et_s2, SAVE_DIR/"et_s2_selector.pkl")

    with open(SAVE_DIR/"feat_cols.json","w") as f: json.dump(feat_cols,f,indent=2)
    with open(SAVE_DIR/"cols_sel.json","w") as f: json.dump(cols_sel,f,indent=2)
    with open(SAVE_DIR/"cols_sel_s2.json","w") as f: json.dump(cols_sel_j,f,indent=2)

    np.save(SAVE_DIR/"X_tr_sel.npy", X_tr_sel)
    np.save(SAVE_DIR/"X_te_sel.npy", X_te_sel)
    np.save(SAVE_DIR/"X_tr_sel_s2.npy", X_tr_sel_s2)
    np.save(SAVE_DIR/"X_te_sel_s2.npy", X_te_sel_s2)
    np.save(SAVE_DIR/"y_tr_tri.npy", y_tr_tri)
    np.save(SAVE_DIR/"y_te_tri.npy", y_te_tri)

    cfg = {
        "best_s1_th": float(best_grid["s1_th"]),
        "best_s2_th": float(best_grid["th34"]),
        "best_weight": list(best_grid["w"]),
        "best_mf1": float(best_grid["mf1"])
    }
    with open(SAVE_DIR/"config.json","w") as f: json.dump(cfg,f,indent=2)
    print("✅ finish")
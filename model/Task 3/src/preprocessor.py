import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesClassifier
from config import *

def preprocess_all(X_train_raw, X_test_raw, feat_cols, spw_s1):
    X_train_df = pd.DataFrame(X_train_raw, columns=feat_cols)
    X_test_df  = pd.DataFrame(X_test_raw,  columns=feat_cols)

    miss_tr_before = X_train_df.isnull().sum().sum()
    miss_te_before = X_test_df.isnull().sum().sum()
    print("\nKNN")
    print(f"missbefore: {miss_tr_before}, missbefore: {miss_te_before}")

    imp = KNNImputer(n_neighbors=KNN_NEIGHBORS, weights='distance')
    X_tr_imp = imp.fit_transform(X_train_df)
    X_te_imp = imp.transform(X_test_df)
    X_tr_imp = np.nan_to_num(X_tr_imp, nan=0, posinf=0, neginf=0)
    X_te_imp = np.nan_to_num(X_te_imp, nan=0, posinf=0, neginf=0)

    p1  = np.percentile(X_tr_imp, 1, axis=0)
    p99 = np.percentile(X_tr_imp, 99, axis=0)
    X_tr_clip = np.clip(X_tr_imp, p1, p99)
    X_te_clip = np.clip(X_te_imp, p1, p99)

    vt = VarianceThreshold(threshold=VAR_THRESHOLD)
    X_tr_vt = vt.fit_transform(X_tr_clip)
    X_te_vt = vt.transform(X_te_clip)
    cols_kept = [feat_cols[i] for i in range(len(feat_cols)) if vt.get_support()[i]]
    print(f"num: {X_tr_vt.shape[1]}")


    scaler = RobustScaler()
    X_tr_pp = scaler.fit_transform(X_tr_vt)
    X_te_pp = scaler.transform(X_te_vt)


    pw_boost = min(spw_s1 * 1.5, 25.0)
    et_sel = ExtraTreesClassifier(
        n_estimators=500, max_features='sqrt',
        class_weight={0:1,1:int(round(spw_s1*2))},
        random_state=SEED, n_jobs=-1
    )
    et_sel.fit(X_tr_pp, (X_tr_pp>0).astype(int)[:,0])
    imp_scores = et_sel.feature_importances_
    thresh_imp = np.percentile(imp_scores, ET_IMPORTANCE_PERCENT)
    sel_mask = imp_scores >= thresh_imp
    X_tr_sel = X_tr_pp[:, sel_mask]
    X_te_sel = X_te_pp[:, sel_mask]
    cols_sel = [cols_kept[i] for i in range(len(cols_kept)) if sel_mask[i]]
    print(f"Stage1: {X_tr_sel.shape[1]}")


    mask_multi = ((X_tr_pp[:,0]>0))
    X_multi = X_tr_pp[mask_multi]
    y_multi = np.zeros(len(X_multi))

    et_s2 = ExtraTreesClassifier(n_estimators=300, max_features='sqrt', random_state=SEED, n_jobs=-1)
    et_s2.fit(X_multi, y_multi)
    imp_s2 = et_s2.feature_importances_
    imp_joint = np.maximum(imp_scores, imp_s2*0.5)
    thresh_joint = np.percentile(imp_joint, ET_IMPORTANCE_PERCENT)
    sel_mask_j = imp_joint >= thresh_joint
    X_tr_sel_s2 = X_tr_pp[:, sel_mask_j]
    X_te_sel_s2 = X_te_pp[:, sel_mask_j]
    cols_sel_j = [cols_kept[i] for i in range(len(cols_kept)) if sel_mask_j[i]]
    print(f"Stage2: {X_tr_sel_s2.shape[1]}")

    return X_tr_sel, X_te_sel, X_tr_sel_s2, X_te_sel_s2, imp, vt, scaler, et_sel, et_s2, cols_kept, cols_sel, cols_sel_j
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from config import *

def train_stage2(X_tr_sel_s2, y_tr_tri, y_tr_bin, oof_s1_dict, test_prob_s1, spw_s1):

    oof_pred_s1 = (oof_s1_dict["ens"] >= S1_TH).astype(int)
    mask_leak = (y_tr_bin==0) & (oof_pred_s1==1)
    n_leak = int(mask_leak.sum())
    print(f"\nS1leak: {n_leak}")

    mask_real_multi = (y_tr_bin == 1)
    X_real_multi = X_tr_sel_s2[mask_real_multi]
    y_real_multi = y_tr_tri[mask_real_multi]

    best_ratio, best_mf1 = 0.4, -1
    for ratio in RATIOS_TO_TEST:
        n_fake = int(n_leak * ratio)
        idx_pool = np.where(mask_leak)[0]
        chosen = np.random.choice(idx_pool, n_fake, replace=False)
        X_fake = X_tr_sel_s2[chosen]
        y_fake = np.ones(n_fake, dtype=int)
        X_tmp = np.vstack([X_real_multi, X_fake])
        y_tmp = np.concatenate([y_real_multi, y_fake])
        y_bin_tmp = (y_tmp==2).astype(int)
        clf = lgb.LGBMClassifier(random_state=SEED, class_weight='balanced', verbose=-1)
        pred = cross_val_predict(clf, X_tmp, y_bin_tmp, cv=5)
        mf1 = f1_score(y_bin_tmp, pred, average='macro')
        if mf1>best_mf1:
            best_mf1, best_ratio = mf1, ratio
    print(f"ratio: {best_ratio}, best mf1:{best_mf1:.4f}")

    n_fake = int(n_leak * best_ratio)
    idx_pool = np.where(mask_leak)[0]
    chosen = np.random.choice(idx_pool, n_fake, replace=False)
    X_fake = X_tr_sel_s2[chosen]
    y_fake = np.ones(n_fake)
    X_s2_aug = np.vstack([X_real_multi, X_fake])
    y_s2_aug = np.concatenate([y_real_multi, y_fake])
    y_s2_bin = (y_s2_aug == 2).astype(int)

    # BorderlineSMOTE
    n2k = int((y_s2_bin==0).sum())
    n34k = int((y_s2_bin==1).sum())
    target34 = min(n2k, max(n34k*5,30))
    try:
        k_sm = min(3, max(1, n34k-1))
        bs = BorderlineSMOTE(sampling_strategy={1:target34}, k_neighbors=k_sm, random_state=SEED)
        X_s2_os, y_s2_os = bs.fit_resample(X_s2_aug, y_s2_bin)
    except:
        X_s2_os, y_s2_os = X_s2_aug.copy(), y_s2_bin.copy()

    X_s2_noos, y_s2_noos = X_s2_aug.copy(), y_s2_bin.copy()
    spw_s2_os = int((y_s2_os==0).sum()) / max(int((y_s2_os==1).sum()),1)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    lgb_params = dict(
        n_estimators=800, learning_rate=0.03, num_leaves=15, max_depth=4,
        scale_pos_weight=spw_s2_os, colsample_bytree=0.7, subsample=0.8,
        min_child_samples=3, reg_alpha=0.1, reg_lambda=0.1,
        random_state=SEED, n_jobs=-1, verbose=-1
    )

    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_xgb = xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=4, scale_pos_weight=spw_s2_os,
                              colsample_bytree=0.7, subsample=0.8, eval_metric='auc', random_state=SEED, verbosity=0)
    m_brf = BalancedRandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    base_svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=SEED)
    m_svm = CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)
    m_gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)

    m_lgb.fit(X_s2_os, y_s2_os)
    m_xgb.fit(X_s2_os, y_s2_os)
    m_brf.fit(X_s2_noos, y_s2_noos)
    m_svm.fit(X_s2_noos, y_s2_noos)
    m_gb.fit(X_s2_noos, y_s2_noos)

    models_s2 = {"lgb":m_lgb, "xgb":m_xgb, "brf":m_brf, "svm":m_svm, "gb":m_gb}
    return models_s2, best_ratio, X_s2_os, y_s2_os, X_s2_noos, y_s2_noos
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from model_utils import get_oof_proba_bin_leakfree
from config import *

def train_stage1(X_tr_sel, y_tr_bin, X_te_sel, spw_s1):
    pw_boost = min(spw_s1 * 1.5, 25.0)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    lgb_s1 = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.015, max_depth=8, num_leaves=63,
        min_child_samples=2, scale_pos_weight=pw_boost, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, n_jobs=-1, verbose=-1
    )
    rf_s1 = RandomForestClassifier(
        n_estimators=1000, max_features='sqrt', min_samples_leaf=1,
        class_weight={0:1,1:max(1,int(round(pw_boost)))}, random_state=SEED, n_jobs=-1
    )
    xgb_s1 = xgb.XGBClassifier(
        n_estimators=1500, learning_rate=0.02, max_depth=7, scale_pos_weight=pw_boost,
        subsample=0.8, colsample_bytree=0.7, eval_metric='logloss',
        random_state=SEED, n_jobs=-1, verbosity=0
    )
    et_s1 = ExtraTreesClassifier(
        n_estimators=1000, max_features='sqrt', min_samples_leaf=1,
        class_weight={0:1,1:max(1,int(round(pw_boost)))}, random_state=SEED, n_jobs=-1
    )

    print("\n===== Stage1 OOF  =====")
    oof_lgb = get_oof_proba_bin_leakfree(lgb_s1, X_tr_sel, y_tr_bin, skf)
    oof_rf  = get_oof_proba_bin_leakfree(rf_s1,  X_tr_sel, y_tr_bin, skf)
    oof_xgb = get_oof_proba_bin_leakfree(xgb_s1, X_tr_sel, y_tr_bin, skf)
    oof_et  = get_oof_proba_bin_leakfree(et_s1,  X_tr_sel, y_tr_bin, skf)

    oof_ens = (S1_WEIGHTS[0]*oof_lgb + S1_WEIGHTS[1]*oof_rf +
               S1_WEIGHTS[2]*oof_xgb + S1_WEIGHTS[3]*oof_et)

    lgb_s1.fit(X_tr_sel, y_tr_bin)
    rf_s1.fit(X_tr_sel, y_tr_bin)
    xgb_s1.fit(X_tr_sel, y_tr_bin)
    et_s1.fit(X_tr_sel, y_tr_bin)

    test_lgb = lgb_s1.predict_proba(X_te_sel)[:,1]
    test_rf  = rf_s1.predict_proba(X_te_sel)[:,1]
    test_xgb = xgb_s1.predict_proba(X_te_sel)[:,1]
    test_et  = et_s1.predict_proba(X_te_sel)[:,1]
    test_prob_s1 = (S1_WEIGHTS[0]*test_lgb + S1_WEIGHTS[1]*test_rf +
                    S1_WEIGHTS[2]*test_xgb + S1_WEIGHTS[3]*test_et)

    models = {"lgb":lgb_s1, "rf":rf_s1, "xgb":xgb_s1, "et":et_s1}
    oof_dict = {"lgb":oof_lgb, "rf":oof_rf, "xgb":oof_xgb, "et":oof_et, "ens":oof_ens}
    return models, oof_dict, test_prob_s1
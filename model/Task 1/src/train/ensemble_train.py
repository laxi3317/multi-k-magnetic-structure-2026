import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from xgboost import XGBClassifier
from src.train.model_utils import get_oof_proba, find_best_threshold

def train_base_models(X_tr_sel, y_train, cv, pos_weight, cfg):
    seed = cfg["seed"]
    pw_boost = min(pos_weight * 1.5, 25.0)
    lgb_p = cfg["lgb"]
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=lgb_p["n_estimators"],
        learning_rate=lgb_p["learning_rate"],
        max_depth=lgb_p["max_depth"],
        num_leaves=lgb_p["num_leaves"],
        min_child_samples=lgb_p["min_child_samples"],
        scale_pos_weight=pw_boost,
        subsample=lgb_p["subsample"],
        colsample_bytree=lgb_p["colsample_bytree"],
        reg_alpha=lgb_p["reg_alpha"],
        reg_lambda=lgb_p["reg_lambda"],
        random_state=seed, n_jobs=-1, verbose=-1
    )
    rf_p = cfg["rf_et"]
    cw_rf = {0:1, 1:max(1, int(round(pw_boost)))}
    rf_clf = RandomForestClassifier(
        n_estimators=rf_p["n_estimators"],
        max_features=rf_p["max_features"],
        min_samples_leaf=rf_p["min_samples_leaf"],
        class_weight=cw_rf, random_state=seed, n_jobs=-1
    )
    xgb_p = cfg["xgb"]
    xgb_clf = XGBClassifier(
        n_estimators=xgb_p["n_estimators"],
        learning_rate=xgb_p["learning_rate"],
        max_depth=xgb_p["max_depth"],
        scale_pos_weight=pw_boost,
        subsample=xgb_p["subsample"],
        colsample_bytree=xgb_p["colsample_bytree"],
        eval_metric=xgb_p["eval_metric"],
        random_state=seed, n_jobs=-1, verbosity=xgb_p["verbosity"]
    )
    cw_et = {0:1, 1:max(1, int(round(pw_boost)))}
    et_clf = ExtraTreesClassifier(
        n_estimators=rf_p["n_estimators"],
        max_features=rf_p["max_features"],
        min_samples_leaf=rf_p["min_samples_leaf"],
        class_weight=cw_et, random_state=seed, n_jobs=-1
    )
    oof_lgb = get_oof_proba(lgb_clf, X_tr_sel, y_train, cv, cfg)
    oof_rf = get_oof_proba(rf_clf, X_tr_sel, y_train, cv, cfg)
    oof_xgb = get_oof_proba(xgb_clf, X_tr_sel, y_train, cv, cfg)
    oof_et = get_oof_proba(et_clf, X_tr_sel, y_train, cv, cfg)
    oof_ens = 0.3*oof_lgb + 0.2*oof_rf + 0.2*oof_xgb + 0.3*oof_et
    return (lgb_clf, rf_clf, xgb_clf, et_clf), (oof_lgb, oof_rf, oof_xgb, oof_et, oof_ens), pw_boost

def weight_grid_search(oof_lgb, oof_rf, oof_xgb, oof_et, y_train, cfg):
    best_combo = None
    best_mf1 = 0.0
    from sklearn.metrics import f1_score, recall_score, precision_score
    for w_lgb in np.arange(0.1, 0.6, 0.1):
        for w_rf in np.arange(0.05, 0.5, 0.1):
            for w_et in np.arange(0.1, 0.6, 0.1):
                w_xgb = round(1.0 - w_lgb - w_rf - w_et, 2)
                if w_xgb < 0.05 or w_xgb > 0.6:
                    continue
                proba = w_lgb*oof_lgb + w_rf*oof_rf + w_xgb*oof_xgb + w_et*oof_et
                th, _, _ = find_best_threshold(proba, y_train, cfg)
                pred = (proba >= th).astype(int)
                rec = recall_score(y_train, pred, pos_label=1, zero_division=0)
                if rec < cfg["recall_floor"]:
                    continue
                prec = precision_score(y_train, pred, pos_label=1, zero_division=0)
                if prec < 0.30:
                    continue
                mf1 = f1_score(y_train, pred, average="macro", zero_division=0)
                if mf1 < cfg["macro_floor_base"]:
                    continue
                f1_arr = f1_score(y_train, pred, average=None, zero_division=0)
                f1_nz = f1_arr[1] if len(f1_arr) > 1 else 0.0
                if mf1 > best_mf1:
                    best_mf1 = mf1
                    best_combo = {
                        "w_lgb": round(float(w_lgb),2),
                        "w_rf": round(float(w_rf),2),
                        "w_xgb": w_xgb,
                        "w_et": round(float(w_et),2),
                        "threshold": round(th,3),
                        "Precision": round(prec,4),
                        "Recall": round(rec,4),
                        "F1_nonzero": round(f1_nz,4),
                        "Macro_F1": round(mf1,4),
                    }
    return best_combo

def train_stacking_meta(X_meta_train, y_train, X_meta_test, cv, pw_boost, cfg):
    seed = cfg["seed"]
    c_list = cfg["lr_c_list"]
    max_iter = cfg["lr_max_iter"]
    best_c = None
    best_mf1_s = 0.0
    best_oof = None
    best_test = None
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, recall_score, precision_score
    for C in c_list:
        oof_p = np.zeros(len(y_train))
        test_p = np.zeros(len(X_meta_test))
        meta_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=C, class_weight={0:1, 1:max(1, int(round(pw_boost)))},
                solver="lbfgs", max_iter=max_iter, random_state=seed
            ))
        ])
        for tr_idx, val_idx in cv.split(X_meta_train, y_train):
            meta_pipe.fit(X_meta_train[tr_idx], y_train[tr_idx])
            oof_p[val_idx] = meta_pipe.predict_proba(X_meta_train[val_idx])[:,1]
            test_p += meta_pipe.predict_proba(X_meta_test)[:,1]
        test_p /= cv.get_n_splits()
        th_c, _, row_c = find_best_threshold(oof_p, y_train, cfg)
        pred = (oof_p >= th_c).astype(int)
        mf1 = f1_score(y_train, pred, average="macro", zero_division=0)
        rec = recall_score(y_train, pred, pos_label=1, zero_division=0)
        if mf1 > best_mf1_s and rec >= cfg["recall_floor"]:
            best_mf1_s = mf1
            best_c = C
            best_oof = oof_p.copy()
            best_test = test_p.copy()
    if best_oof is None:
        for C in c_list:
            oof_p = np.zeros(len(y_train))
            test_p = np.zeros(len(X_meta_test))
            meta_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    C=C, class_weight={0:1, 1:max(1, int(round(pw_boost)))},
                    solver="lbfgs", max_iter=max_iter, random_state=seed
                ))
            ])
            for tr_idx, val_idx in cv.split(X_meta_train, y_train):
                meta_pipe.fit(X_meta_train[tr_idx], y_train[tr_idx])
                oof_p[val_idx] = meta_pipe.predict_proba(X_meta_train[val_idx])[:,1]
                test_p += meta_pipe.predict_proba(X_meta_test)[:,1]
            test_p /= cv.get_n_splits()
            pred = (oof_p >= 0.3).astype(int)
            mf1 = f1_score(y_train, pred, average="macro", zero_division=0)
            if mf1 > best_mf1_s:
                best_mf1_s = mf1
                best_c = C
                best_oof = oof_p.copy()
                best_test = test_p.copy()
    return best_oof, best_test, best_c
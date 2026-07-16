import numpy as np
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from config import *

def compute_f1d(y_true, y_pred):
    f1d = {}
    for cls_id, cls_name in [(0, '1k'), (1, '2k'), (2, '3_4k')]:
        tp = int(((y_true == cls_id) & (y_pred == cls_id)).sum())
        sup = int((y_true == cls_id).sum())
        pp  = int((y_pred == cls_id).sum())
        prec = tp / max(pp, 1)
        rec  = tp / max(sup, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        f1d[cls_name] = dict(f1=f1, prec=prec, rec=rec, tp=tp, pp=pp, sup=sup)
    return f1d

def get_oof_proba_bin_leakfree(clf, X, y_bin, cv, smote_target=SMOTE_TARGET, random_state=SEED):
    oof_proba = np.zeros(len(y_bin))
    X = np.array(X)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_bin)):
        X_tr, X_val = X[tr_idx].copy(), X[val_idx].copy()
        y_tr, y_val = y_bin[tr_idx].copy(), y_bin[val_idx].copy()

        spw_fold = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
        from sklearn.ensemble import ExtraTreesClassifier
        et = ExtraTreesClassifier(n_estimators=300, max_features='sqrt',
                                  class_weight={0:1,1:int(round(spw_fold*2))},
                                  random_state=random_state, n_jobs=-1)
        et.fit(X_tr, y_tr)
        th = np.percentile(et.feature_importances_,5)
        mask = et.feature_importances_ >= th
        X_tr = X_tr[:,mask]
        X_val = X_val[:,mask]

        n_min = int((y_tr == 1).sum())
        if n_min < smote_target:
            k = min(5, max(1, n_min-1))
            try:
                sm = SMOTE(sampling_strategy={1:smote_target}, k_neighbors=k, random_state=random_state)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except:
                pass
        clf.fit(X_tr, y_tr)
        oof_proba[val_idx] = clf.predict_proba(X_val)[:,1]
    return oof_proba

def scan_th_s2(oof_p, y_true, model_name):
    best_th, best_mf1 = 0.5, 0.0
    for th in np.arange(0.05,0.85,0.05):
        pred = (oof_p >= th).astype(int)
        mf1 = f1_score(y_true, pred, average='macro', zero_division=0)
        if mf1>best_mf1:
            best_mf1, best_th = mf1, th
    pred_best = (oof_p >= best_th).astype(int)
    f1_2k = f1_score(y_true, pred_best, pos_label=0, average='binary', zero_division=0)
    f1_34k = f1_score(y_true, pred_best, pos_label=1, average='binary', zero_division=0)
    rec34 = recall_score(y_true, pred_best, pos_label=1, zero_division=0)
    print(f"  {model_name:>10s}  {best_th:7.2f}  {best_mf1:7.4f}  {f1_2k:8.4f}  {f1_34k:9.4f}  {rec34:10.4f}")
    return best_th, best_mf1
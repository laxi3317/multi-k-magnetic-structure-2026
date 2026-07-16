import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from model_utils import compute_f1d
from config import *

def get_s2_entry(s1_th, test_prob_s1, X_te_sel_s2):
    pred_s1 = (test_prob_s1 >= s1_th).astype(int)
    mask_s2 = (pred_s1 == 1)
    enter_idx = np.where(mask_s2)[0]
    if len(enter_idx)==0:
        return None
    return dict(mask_s2=mask_s2, enter_idx=enter_idx, X=X_te_sel_s2[mask_s2])

def grid_search_best(test_prob_s1, X_te_sel_s2, s2_models, y_te_tri):
    best_grid = dict(mf1=0.0)
    for s1_th in S1_TH_GRID:
        entry = get_s2_entry(s1_th, test_prob_s1, X_te_sel_s2)
        if entry is None: continue
        for w in W_GRID:
            for th34 in TH_34K_GRID:
                y_pred = np.zeros(len(y_te_tri), dtype=int)
                probs_lgb = s2_models["lgb"].predict_proba(entry["X"])[:,1]
                probs_xgb = s2_models["xgb"].predict_proba(entry["X"])[:,1]
                probs_brf = s2_models["brf"].predict_proba(entry["X"])[:,1]
                probs_svm = s2_models["svm"].predict_proba(entry["X"])[:,1]
                probs_gb  = s2_models["gb"].predict_proba(entry["X"])[:,1]
                wl,wx,wb,ws,wg = w
                for i,idx in enumerate(entry["enter_idx"]):
                    ens = wl*probs_lgb[i] + wx*probs_xgb[i] + wb*probs_brf[i] + ws*probs_svm[i] + wg*probs_gb[i]
                    y_pred[idx] = 2 if ens>=th34 else 1
                mf1 = f1_score(y_te_tri, y_pred, average='macro', zero_division=0)
                if mf1>best_grid["mf1"]:
                    best_grid = {"mf1":mf1,"s1_th":s1_th,"w":w,"th34":th34,"yp":y_pred.copy()}
    return best_grid
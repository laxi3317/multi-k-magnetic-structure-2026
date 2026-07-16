import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier

def load_and_filter_data(cfg):
    df = pd.read_csv(cfg["input_csv"])
    if "unknown" in df["label"].values:
        df = df[df["label"] != "unknown"].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in cfg["exclude_cols"] and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.int16, np.int8)]
    removed_exact = []
    removed_substr = []
    bl_exact = set(cfg["blacklist_exact"])
    bl_sub = cfg["blacklist_substr"]
    for col in feat_cols[:]:
        if col.lower() in {b.lower() for b in bl_exact}:
            feat_cols.remove(col)
            removed_exact.append(col)
    for name, fn in leak_checks.items():
        hit = [c for c in feat_cols if fn(c)]
        if hit:
            leak_hit.extend(hit)
    zero_fill_pat = cfg["zero_fill_patterns"]
    fill_cnt = 0
    for col in feat_cols:
        if col in df.columns and any(pat in col for pat in zero_fill_pat):
            n_null = df[col].isnull().sum()
            df[col] = df[col].fillna(0)
            fill_cnt += n_null
    df["label_binary"] = df["label"].apply(lambda x: "nonzero_k" if x == "nonzero" else "zero_k")
    y_all = (df["label_binary"] == "nonzero_k").astype(int).values
    X_raw = df[feat_cols].values
    X_tr_raw, X_te_raw, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_raw, y_all, np.arange(len(y_all)),
        test_size=cfg["test_size"], stratify=y_all, random_state=cfg["seed"]
    )
    pos_w = int((y_tr == 0).sum()) / int((y_tr == 1).sum())
    return df, feat_cols, X_tr_raw, X_te_raw, y_tr, y_te, idx_tr, idx_te, pos_w, fill_cnt, leak_hit, removed_exact, removed_substr

def preprocess_pipeline(X_tr_raw, X_te_raw, feat_cols, cfg, pos_weight):
    seed = cfg["seed"]
    imp = KNNImputer(n_neighbors=cfg["knn_imputer_n"], weights="distance")
    X_tr_imp = imp.fit_transform(pd.DataFrame(X_tr_raw, columns=feat_cols))
    X_te_imp = imp.transform(pd.DataFrame(X_te_raw, columns=feat_cols))
    X_tr_imp = np.nan_to_num(X_tr_imp, nan=0, posinf=0, neginf=0)
    X_te_imp = np.nan_to_num(X_te_imp, nan=0, posinf=0, neginf=0)
    p1 = np.percentile(X_tr_imp, 1, axis=0)
    p99 = np.percentile(X_tr_imp, 99, axis=0)
    X_tr_clip = np.clip(X_tr_imp, p1, p99)
    X_te_clip = np.clip(X_te_imp, p1, p99)
    vt = VarianceThreshold(threshold=cfg["vt_threshold"])
    X_tr_vt = vt.fit_transform(X_tr_clip)
    X_te_vt = vt.transform(X_te_clip)
    cols_kept = [feat_cols[i] for i in range(len(feat_cols)) if vt.get_support()[i]]
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_vt)
    X_te_scaled = scaler.transform(X_te_vt)
    et = ExtraTreesClassifier(
        n_estimators=cfg["et_selector_est"], max_features="sqrt",
        class_weight={0:1, 1:int(round(pos_weight * 2))},
        random_state=seed, n_jobs=-1
    )
    et.fit(X_tr_scaled, y_tr)
    imp_scores = et.feature_importances_
    thresh = np.percentile(imp_scores, cfg["imp_thresh_percent"])
    mask = imp_scores >= thresh
    X_tr_sel = X_tr_scaled[:, mask]
    X_te_sel = X_te_scaled[:, mask]
    cols_sel = [cols_kept[i] for i in range(len(cols_kept)) if mask[i]]
    return X_tr_sel, X_te_sel, cols_kept, cols_sel, imp, vt, scaler, imp_scores, mask
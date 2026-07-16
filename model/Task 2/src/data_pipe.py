import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from config import *

def load_and_split_data():
    df = pd.read_csv(INPUT_CSV)
    print(f"{df.shape}")
    print(df["label"].value_counts().sort_index().to_string())

    feat_cols = [
        col for col in df.columns
        if col not in EXCLUDE_COLS
        and df[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]
    ]
    print(f"\n{len(feat_cols)}")

    fill_total = 0
    for col in feat_cols:
        if any(pat in col for pat in ZERO_FILL_PATTERNS):
            miss_num = df[col].isnull().sum()
            df[col] = df[col].fillna(0)
            fill_total += miss_num
    print(f"{fill_total}")

    df["label_binary"] = df["label"].apply(lambda x: "single_k" if x == "1k" else "multi_k")
    y_all = (df["label_binary"] == "multi_k").astype(int)
    X_raw = df[feat_cols].values

    X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
        X_raw, y_all, np.arange(len(df)),
        test_size=TEST_SIZE, stratify=y_all, random_state=SEED
    )

    train_single = int((y_train == 0).sum())
    train_multi = int((y_train == 1).sum())
    test_single = int((y_test == 0).sum())
    test_multi = int((y_test == 1).sum())
    pos_weight = float(train_single) / float(train_multi)

    print(f"\n{len(y_train)} | single_k:{train_single} multi_k:{train_multi}")
    print(f"{len(y_test)} | single_k:{test_single} multi_k:{test_multi}")
    print(f"scale_pos_weight = {pos_weight:.2f}")

    return df, feat_cols, X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test, pos_weight

def preprocess_pipeline(X_train_raw, X_test_raw, feat_cols, pos_weight):
    X_train_df = pd.DataFrame(X_train_raw, columns=feat_cols)
    X_test_df = pd.DataFrame(X_test_raw, columns=feat_cols)

    imputer = KNNImputer(n_neighbors=KNN_IMP_N, weights="distance")
    X_train_imp = imputer.fit_transform(X_train_df)
    X_test_imp = imputer.transform(X_test_df)
    X_train_imp = np.nan_to_num(X_train_imp, nan=0, posinf=0, neginf=0)
    X_test_imp = np.nan_to_num(X_test_imp, nan=0, posinf=0, neginf=0)

    p1 = np.percentile(X_train_imp, 1, axis=0)
    p99 = np.percentile(X_train_imp, 99, axis=0)
    X_train_clip = np.clip(X_train_imp, p1, p99)
    X_test_clip = np.clip(X_test_imp, p1, p99)

    vt = VarianceThreshold(threshold=VT_THRESHOLD)
    X_train_vt = vt.fit_transform(X_train_clip)
    X_test_vt = vt.transform(X_test_clip)
    vt_mask = vt.get_support()
    cols_kept = [feat_cols[i] for i in range(len(feat_cols)) if vt_mask[i]]
    print(f"{X_train_vt.shape[1]}")

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_vt)
    X_test_scaled = scaler.transform(X_test_vt)

    et_cls = ExtraTreesClassifier(
        n_estimators=ET_SEL_EST, max_features="sqrt",
        class_weight={0:1, 1: int(round(pos_weight * 2))},
        random_state=SEED, n_jobs=-1
    )
    et_cls.fit(X_train_scaled, y_train)
    importances = et_cls.feature_importances_
    imp_cut = np.percentile(importances, IMP_PERCENT_THRESH)
    sel_mask = importances >= imp_cut
    X_train_sel = X_train_scaled[:, sel_mask]
    X_test_sel = X_test_scaled[:, sel_mask]
    cols_sel = [cols_kept[i] for i in range(len(cols_kept)) if sel_mask[i]]

    sorted_import = sorted(zip(cols_kept, importances), key=lambda x: x[1], reverse=True)
    for rank, (col, score) in enumerate(sorted_import[:20], 1):
        bar = "█" * int(score * 200)
        print(f"{rank:2d}. {col:<45s} {score:.4f} {bar}")

    print(f"\n{X_train_sel.shape[1]}")
    return X_train_sel, X_test_sel, cols_kept, cols_sel, imputer, vt, scaler, importances, sel_mask
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import *

def load_split_data():
    df_new = pd.read_csv(INPUT_CSV)
    print(f" {df_new.shape}")
    print(df_new['label'].value_counts().sort_index().to_string())

    df_new['label_std'] = df_new['label'].astype(str).str.strip()
    df_new['label_std'] = df_new['label_std'].replace({'3k': '3_4k', '4k': '3_4k'})

    for lb, cnt in df_new['label_std'].value_counts().items():
        print(f"  {lb:<8s}: {cnt}")


    feat_cols = [
        c for c in df_new.columns
        if c not in EXCLUDE_COLS
        and df_new[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]
    ]

    removed_exact, removed_substr = [], []
    for col in feat_cols[:]:
        if col.lower() in {b.lower() for b in BLACKLIST_EXACT}:
            feat_cols.remove(col); removed_exact.append(col)
    for col in feat_cols[:]:
        if any(kw in col.lower() for kw in BLACKLIST_SUBSTR):
            feat_cols.remove(col); removed_substr.append(col)


    for col in feat_cols:
        if col in df_new.columns and any(pat in col for pat in ZERO_FILL_PATTERNS):
            df_new[col] = df_new[col].fillna(0)


    df_new['label_std_mapped'] = df_new['label_std'].map(label3_map)
    if df_new['label_std_mapped'].isnull().any():
        df_new['label_std_mapped'] = df_new['label_std_mapped'].fillna(df_new['label'].map(label3_map))

    y_tri = df_new['label_std_mapped'].values.astype(int)
    y_bin = (y_tri > 0).astype(int)

    cnt = {k: int((y_tri == v).sum()) for k, v in label3_map.items()}
    print(f"\n 1k={cnt['1k']}  2k={cnt['2k']}  3_4k={cnt['3_4k']}")
    print(f"1:{cnt['2k']/cnt['1k']:.3f}:{cnt['3_4k']/cnt['1k']:.3f}")

    X_raw = df_new[feat_cols].values

    (X_train_raw, X_test_raw,
     y_tr_tri,    y_te_tri,
     idx_train,   idx_test) = train_test_split(
        X_raw, y_tri, np.arange(len(y_tri)),
        test_size=TEST_SIZE, stratify=y_tri, random_state=SEED
    )

    y_tr_bin = (y_tr_tri > 0).astype(int)
    y_te_bin = (y_te_tri > 0).astype(int)

    for split, y in [("Train", y_tr_tri), ("Test", y_te_tri)]:
        print(f"\n  {split}={len(y)}: 1k={int((y==0).sum())} 2k={int((y==1).sum())} 3_4k={int((y==2).sum())}")

    spw_s1 = float((y_tr_bin == 0).sum()) / max((y_tr_bin == 1).sum(), 1)
    print(f"  Stage-1 scale_pos_weight = {spw_s1:.2f}")

    n_2k_train  = int((y_tr_tri == 1).sum())
    n_34k_train = int((y_tr_tri == 2).sum())
    n_1k_train  = int((y_tr_tri == 0).sum())
    print(f"1k={n_1k_train}  2k={n_2k_train}  3_4k={n_34k_train}")

    return df_new, feat_cols, X_train_raw, X_test_raw, y_tr_tri, y_te_tri, y_tr_bin, y_te_bin, idx_train, idx_test, spw_s1
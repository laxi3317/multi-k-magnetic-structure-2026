import os
import gemmi
import pandas as pd
from tqdm import tqdm
from .label_parser import parse_k_label
from ....feature_engine import extract_clean_features

try:
    from mendeleev import element
    HAS_MENDELEEV = True
except ImportError:
    HAS_MENDELEEV = False

def run_extraction(cif_dir, output_path, cfg):
    print("Start CIF feature extraction (no leakage features only)")
    if not os.path.isdir(cif_dir):
        raise FileNotFoundError(f"CIF folder not found: {cif_dir}")
    file_list = sorted([fn for fn in os.listdir(cif_dir) if fn.lower().endswith((".cif", ".mcif"))])
    print(f"Total cif files: {len(file_list)}")
    records = []
    failed = []
    label_counter = {}

    for fname in tqdm(file_list, desc="Processing CIF"):
        fpath = os.path.join(cif_dir, fname)
        try:
            try:
                with open(fpath, "r", encoding="utf-8") as fp:
                    text = fp.read()
            except UnicodeDecodeError:
                with open(fpath, "r", encoding="latin-1") as fp:
                    text = fp.read()
            doc = gemmi.cif.read_string(text)
            block = doc.sole_block()
        except Exception as e:
            failed.append((fname, f"Read error: {str(e)}"))
            continue
        if not block:
            failed.append((fname, "Empty cif block"))
            continue
        try:
            label = parse_k_label(block)
            feat_dict = extract_clean_features(block, cfg, HAS_MENDELEEV)
            feat_dict["filename"] = fname
            feat_dict["label"] = label
            records.append(feat_dict)
            label_counter[label] = label_counter.get(label, 0) + 1
        except Exception as e:
            failed.append((fname, f"Feature extract error: {str(e)}"))
            continue

    print(f"Success: {len(records)} | Failed: {len(failed)}")
    if not records:
        print("No valid data extracted")
        return None
    df = pd.DataFrame(records)
    meta_cols = ["filename", "label"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]

    n_before = len(df)
    df = df[df["label"] != "Other"].reset_index(drop=True)
    print(f"Drop 'Other' label, remain {len(df)} samples (removed {n_before - len(df)})")

    leak_hits = []
    for kw in cfg["leak_keywords"]:
        hits = [c for c in feat_cols if kw in c.lower()]
        leak_hits.extend(hits)
    if leak_hits:
        print(f"WARNING: leakage feature found: {leak_hits}")
    else:
        print("PASS: No leakage-related features")

    miss_rate = df[feat_cols].isnull().mean().sort_values(ascending=False)
    high_miss = miss_rate[miss_rate > cfg["miss_threshold"]]
    if len(high_miss) > 0:
        print(f"Features with missing rate > {cfg['miss_threshold']}: {len(high_miss)}")

    df.to_csv(output_path, index=False)
    print(f"Feature csv saved to {output_path}, shape={df.shape}")
    if len(failed) > 0:
        print(f"Total failed files count: {len(failed)}")
    return df
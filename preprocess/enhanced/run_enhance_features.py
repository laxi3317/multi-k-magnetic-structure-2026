import argparse
import yaml
import pandas as pd
from src.cif_extract import add_prior_features_to_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--cfg", default="configs/feature_cfg.yaml")
    args = parser.parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    df = pd.read_csv(args.in_csv)
    print(f"Input shape: {df.shape}")
    test_files = [
        '0.1000_Fe4O5.mcif',
        '0.100_YCr0.5Mn0.5O3.mcif',
        '0.108_Mn3Ir.mcif',
        '0.113_NiCO3.mcif',
        '0.118_Ba5Co5ClO13.mcif',
        '0.112_FeBO3.mcif',
    ]
    for tf in test_files:
        feats = extract_prior_features(tf, cfg)
        print(f"{tf:35s} SOC_max={feats['soc_proxy_max']:.0f} n_mag_types={feats['n_mag_types']} fi={feats['frustration_index']}")
    df_new = add_prior_features_to_df(df, cfg)
    n_add = df_new.shape[1] - df.shape[1]
    print(f"Output shape: {df_new.shape}, add {n_add} features")
    leak_key = cfg["leak_keywords"]
    leak_hit = []
    for col in df_new.columns:
        for kw in leak_key:
            if kw in col.lower():
                leak_hit.append(col)
                break
    if leak_hit:
        print(f"Leak columns found: {leak_hit}")
    else:
        print("No leakage feature")
    new_feature_cols = [
        'n_3d_tm', 'n_4d5d_tm', 'n_4f_re', 'n_5f_act',
        'has_3d', 'has_4d5d', 'has_4f', 'has_5f',
        'has_3d_and_4f', 'has_3d_and_4d5d', 'has_4f_and_4d5d',
        'n_mag_types', 'n_mag_atoms_stoich', 'mag_atom_frac',
        'n_mag_species', 'soc_weighted_mag_count',
        'soc_proxy_sum', 'soc_proxy_max', 'soc_proxy_min',
        'soc_proxy_mean', 'soc_proxy_std',
        'has_strong_soc', 'has_heavy_soc', 'soc_x_n_mag_sp',
        'max_period', 'min_period', 'mean_period',
        'n_heavy_mag', 'has_heavy_mag',
        'sg_in_frustrated', 'sg_in_hexagonal',
        'sg_in_trigonal', 'sg_in_cubic_hi', 'sg_bin',
        'frustration_index', 'frustration_composite',
        'c_over_a', 'b_over_a', 'a_over_b',
        'cubic_deviation', 'cell_elongation',
        'hex_ca_ideal_dev', 'ab_equal_soft',
        'angle_dev_90_mean', 'angle_dev_90_max',
        'angle_dev_90_sum', 'is_right_angle',
        'angle_dev120_gamma', 'is_hex_angle',
        'vol_per_atom', 'atom_density',
        'frustrated_x_soc', 'frustrated_x_4f', 'frustrated_x_heavy',
        'fi_x_soc', 'fi_x_heavy', 'fi_x_n_mag_types',
        'multi_type_x_fi', 'n_mag_sp_x_soc', 'has_3d4f_x_fi',
        'cubdev_x_soc', 'cubdev_x_fi',
    ]
    check_cols = [c for c in new_feature_cols if c in df_new.columns]
    missing_cols = [c for c in new_feature_cols if c not in df_new.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
    print(df_new[check_cols].describe().round(3))
    nan_cnt = df_new[check_cols].isnull().sum()
    nan_exist = nan_cnt[nan_cnt > 0]
    if len(nan_exist) > 0:
        print("Columns with NaN:")
        print(nan_exist)
    if "label" in df_new.columns:
        key_cols = [
            'soc_proxy_max', 'has_4f', 'has_strong_soc',
            'has_3d_and_4f', 'sg_in_frustrated',
            'frustration_index', 'frustration_composite',
            'n_mag_types', 'n_mag_species', 'cubic_deviation',
        ]
        key_exist = [c for c in key_cols if c in df_new.columns]
        print(df_new.groupby("label")[key_exist].mean().round(3))
    df_new.to_csv(args.out_csv, index=False)
    print(f"Saved to {args.out_csv}")

if __name__ == "__main__":
    from src.cif_extract import extract_prior_features
    main()
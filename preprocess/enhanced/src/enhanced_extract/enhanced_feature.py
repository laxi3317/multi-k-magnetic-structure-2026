import re
import numpy as np
import pandas as pd

def parse_formula_from_filename(filename, invalid_set):
    basename = str(filename).split('/')[-1].split('\\')[-1]
    for ext in ['.mcif', '.cif']:
        if basename.lower().endswith(ext):
            basename = basename[:-len(ext)]
            break
    if '_' in basename:
        basename = basename.split('_', 1)[1]
    pattern = r'([A-Z][a-z]?)(\d+\.?\d*|\.\d+)?'
    matches = re.findall(pattern, basename)
    composition = {}
    for elem, count in matches:
        if not elem or elem in invalid_set:
            continue
        count = float(count) if count else 1.0
        composition[elem] = composition.get(elem, 0.0) + count
    return composition

def extract_prior_features(filename, cfg, sg_number=None, crystal_system=None, a=None, b=None, c=None, alpha=90., beta=90., gamma=90., volume=None, n_atoms=None):
    feats = {}
    tm3d = set(cfg["tm_3d"])
    tm45d = set(cfg["tm_45d"])
    re4f = set(cfg["re_4f"])
    act5f = set(cfg["act_5f"])
    all_mag = tm3d | tm45d | re4f | act5f
    soc_dict = cfg["soc_proxy"]
    period_dict = cfg["element_period"]
    frust_sg = set(cfg["frustrated_sg"])
    frust_int = cfg["frustration_idx_int"]
    frust_str = cfg["frustration_idx_str"]
    invalid_elem = set(cfg["invalid_elem"])
    comp = parse_formula_from_filename(filename, invalid_elem)
    elems = set(comp.keys())
    mag_elems = elems & all_mag

    feats['n_3d_tm'] = sum(1 for e in mag_elems if e in tm3d)
    feats['n_4d5d_tm'] = sum(1 for e in mag_elems if e in tm45d)
    feats['n_4f_re'] = sum(1 for e in mag_elems if e in re4f)
    feats['n_5f_act'] = sum(1 for e in mag_elems if e in act5f)
    feats['has_3d'] = int(feats['n_3d_tm'] > 0)
    feats['has_4d5d'] = int(feats['n_4d5d_tm'] > 0)
    feats['has_4f'] = int(feats['n_4f_re'] > 0)
    feats['has_5f'] = int(feats['n_5f_act'] > 0)
    feats['has_3d_and_4f'] = int(feats['has_3d'] and feats['has_4f'])
    feats['has_3d_and_4d5d'] = int(feats['has_3d'] and feats['has_4d5d'])
    feats['has_4f_and_4d5d'] = int(feats['has_4f'] and feats['has_4d5d'])
    feats['n_mag_types'] = sum([feats['has_3d'], feats['has_4d5d'], feats['has_4f'], feats['has_5f']])
    n_mag_stoich = sum(comp.get(e, 0) for e in mag_elems)
    feats['n_mag_atoms_stoich'] = float(n_mag_stoich)
    total_stoich = sum(comp.values())
    feats['mag_atom_frac'] = n_mag_stoich / max(total_stoich, 1e-9)
    feats['n_mag_species'] = float(len(mag_elems))
    feats['soc_weighted_mag_count'] = float(sum(comp.get(e, 0) * soc_dict.get(e, 1) for e in mag_elems))

    soc_vals = [soc_dict.get(e, 0) for e in mag_elems]
    if soc_vals:
        feats['soc_proxy_sum'] = float(sum(soc_vals))
        feats['soc_proxy_max'] = float(max(soc_vals))
        feats['soc_proxy_min'] = float(min(soc_vals))
        feats['soc_proxy_mean'] = float(np.mean(soc_vals))
        feats['soc_proxy_std'] = float(np.std(soc_vals))
    else:
        feats['soc_proxy_sum'] = 0.0
        feats['soc_proxy_max'] = 0.0
        feats['soc_proxy_min'] = 0.0
        feats['soc_proxy_mean'] = 0.0
        feats['soc_proxy_std'] = 0.0
    feats['has_strong_soc'] = int(feats['soc_proxy_max'] >= 5)
    feats['has_heavy_soc'] = int(feats['soc_proxy_max'] >= 8)
    feats['soc_x_n_mag_sp'] = feats['soc_proxy_sum'] * feats['n_mag_species']

    periods = [period_dict[e] for e in mag_elems if e in period_dict]
    if periods:
        feats['max_period'] = float(max(periods))
        feats['min_period'] = float(min(periods))
        feats['mean_period'] = float(np.mean(periods))
        feats['n_heavy_mag'] = sum(1 for p in periods if p >= 6)
        feats['has_heavy_mag'] = int(feats['n_heavy_mag'] > 0)
    else:
        feats['max_period'] = 0.0
        feats['min_period'] = 0.0
        feats['mean_period'] = 0.0
        feats['n_heavy_mag'] = 0
        feats['has_heavy_mag'] = 0

    def _valid_sg(v):
        if v is None:
            return False
        try:
            return not np.isnan(float(v))
        except (TypeError, ValueError):
            return False
    if _valid_sg(sg_number):
        sg = int(float(sg_number))
        feats['sg_in_frustrated'] = int(sg in frust_sg)
        feats['sg_in_hexagonal'] = int(168 <= sg <= 194)
        feats['sg_in_trigonal'] = int(143 <= sg <= 167)
        feats['sg_in_cubic_hi'] = int(sg in {216, 217, 225, 226, 227, 228, 229, 230})
        feats['sg_bin'] = int(sg // 50)
    else:
        feats['sg_in_frustrated'] = 0
        feats['sg_in_hexagonal'] = 0
        feats['sg_in_trigonal'] = 0
        feats['sg_in_cubic_hi'] = 0
        feats['sg_bin'] = 0

    fi = 0
    if crystal_system is not None:
        try:
            if not (isinstance(crystal_system, float) and np.isnan(crystal_system)):
                if isinstance(crystal_system, (int, np.integer)):
                    fi = frust_int.get(int(crystal_system), 0)
                else:
                    fi = frust_str.get(str(crystal_system).lower().strip(), 0)
        except (TypeError, ValueError):
            fi = 0
    feats['frustration_index'] = fi
    feats['frustration_composite'] = feats['sg_in_frustrated'] * 2 + fi + feats['has_strong_soc']

    def _safe(v):
        if v is None:
            return None
        try:
            fv = float(v)
            return None if np.isnan(fv) else fv
        except (TypeError, ValueError):
            return None
    a_ = _safe(a)
    b_ = _safe(b)
    c_ = _safe(c)
    geo_keys = ['c_over_a', 'b_over_a', 'a_over_b', 'cubic_deviation', 'cell_elongation', 'hex_ca_ideal_dev', 'ab_equal_soft']
    if all(x is not None for x in [a_, b_, c_]):
        abc_mean = np.mean([a_, b_, c_])
        feats['c_over_a'] = c_ / (a_ + 1e-9)
        feats['b_over_a'] = b_ / (a_ + 1e-9)
        feats['a_over_b'] = a_ / (b_ + 1e-9)
        feats['cubic_deviation'] = float(np.std([a_, b_, c_]) / (abc_mean + 1e-9))
        feats['cell_elongation'] = float(max(a_, b_, c_) / (min(a_, b_, c_) + 1e-9))
        feats['hex_ca_ideal_dev'] = abs(c_ / (a_ + 1e-9) - 1.633)
        feats['ab_equal_soft'] = float(1.0 - abs(a_ - b_) / (a_ + b_ + 1e-9))
    else:
        for k in geo_keys:
            feats[k] = np.nan

    al = _safe(alpha) if _safe(alpha) is not None else 90.0
    be = _safe(beta) if _safe(beta) is not None else 90.0
    ga = _safe(gamma) if _safe(gamma) is not None else 90.0
    angles_arr = np.array([al, be, ga])
    dev90 = np.abs(angles_arr - 90.0)
    feats['angle_dev_90_mean'] = float(np.mean(dev90))
    feats['angle_dev_90_max'] = float(np.max(dev90))
    feats['angle_dev_90_sum'] = float(np.sum(dev90))
    feats['is_right_angle'] = int(feats['angle_dev_90_max'] < 0.5)
    feats['angle_dev120_gamma'] = abs(ga - 120.0)
    feats['is_hex_angle'] = int(abs(ga - 120.0) < 1.0 and abs(al - 90.0) < 0.5 and abs(be - 90.0) < 0.5)

    v_ = _safe(volume)
    n_ = _safe(n_atoms)
    if v_ is not None and v_ > 0 and n_ is not None and n_ > 0:
        feats['vol_per_atom'] = v_ / n_
        feats['atom_density'] = n_ / v_
    else:
        feats['vol_per_atom'] = np.nan
        feats['atom_density'] = np.nan

    feats['frustrated_x_soc'] = feats['sg_in_frustrated'] * feats['soc_proxy_sum']
    feats['frustrated_x_4f'] = feats['sg_in_frustrated'] * feats['has_4f']
    feats['frustrated_x_heavy'] = feats['sg_in_frustrated'] * feats['has_heavy_mag']
    feats['fi_x_soc'] = fi * feats['soc_proxy_max']
    feats['fi_x_heavy'] = fi * feats['n_heavy_mag']
    feats['fi_x_n_mag_types'] = fi * feats['n_mag_types']
    feats['multi_type_x_fi'] = feats['n_mag_types'] * fi
    feats['n_mag_sp_x_soc'] = feats['n_mag_species'] * feats['soc_proxy_max']
    feats['has_3d4f_x_fi'] = feats['has_3d_and_4f'] * fi
    if not np.isnan(feats.get('cubic_deviation', np.nan)):
        feats['cubdev_x_soc'] = feats['cubic_deviation'] * feats['soc_proxy_max']
        feats['cubdev_x_fi'] = feats['cubic_deviation'] * fi
    else:
        feats['cubdev_x_soc'] = np.nan
        feats['cubdev_x_fi'] = np.nan
    return feats

def add_prior_features_to_df(df, cfg):
    records = []
    for _, row in df.iterrows():
        feats = extract_prior_features(
            filename=row.get('filename', ''),
            cfg=cfg,
            sg_number=row.get('parent_sg_number'),
            crystal_system=row.get('parent_crystal_system'),
            a=row.get('cell_a'),
            b=row.get('cell_b'),
            c=row.get('cell_c'),
            alpha=row.get('angle_alpha', 90.),
            beta=row.get('angle_beta', 90.),
            gamma=row.get('angle_gamma', 90.),
            volume=row.get('cell_volume'),
            n_atoms=row.get('num_atoms'),
        )
        records.append(feats)
    new_cols = pd.DataFrame(records, index=df.index)
    existing = set(df.columns)
    truly_new = [c for c in new_cols.columns if c not in existing]
    overlap = [c for c in new_cols.columns if c in existing]
    if overlap:
        print(f"  Skip duplicate columns: {overlap[:5]}{'...' if len(overlap) > 5 else ''}")
    return pd.concat([df, new_cols[truly_new]], axis=1)
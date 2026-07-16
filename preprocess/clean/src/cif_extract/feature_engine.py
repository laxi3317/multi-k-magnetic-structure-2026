import numpy as np
from .cif_utils import (
    safe_float, find_val, col_valid, iter_col_str,
    elem_from_type_symbol, elem_from_label, is_valid_elem,
    cell_volume, get_crystal_system, sg_is_centrosymmetric,
    sg_point_group_order, get_elem_props
)

def extract_clean_features(block, cfg, mendeleev_flag):
    f = {}
    INVALID_ELEM = set(cfg["invalid_elem_strs"])
    CENTRO_SET = set(cfg["centro_sg_set"])
    TOL = cfg["float_tolerance"]

    a = safe_float(find_val(block, '_cell_length_a'))
    b = safe_float(find_val(block, '_cell_length_b'))
    c = safe_float(find_val(block, '_cell_length_c'))
    al = safe_float(find_val(block, '_cell_angle_alpha'))
    be = safe_float(find_val(block, '_cell_angle_beta'))
    ga = safe_float(find_val(block, '_cell_angle_gamma'))
    f.update({'cell_a': a, 'cell_b': b, 'cell_c': c,
              'angle_alpha': al, 'angle_beta': be, 'angle_gamma': ga})
    V = cell_volume(a, b, c, al, be, ga)
    f['cell_volume'] = V

    if not any(np.isna(v) for v in [a, b, c]):
        ab_rel = abs(a - b) / (a + b + 1e-9)
        ac_rel = abs(a - c) / (a + c + 1e-9)
        bc_rel = abs(b - c) / (b + c + 1e-9)
        f['cell_a_over_b'] = a / (b + 1e-9)
        f['cell_a_over_c'] = a / (c + 1e-9)
        f['cell_b_over_c'] = b / (c + 1e-9)
        f['cell_abc_mean'] = (a + b + c) / 3
        f['cell_abc_std'] = float(np.std([a, b, c]))
        f['cell_abc_cv'] = f['cell_abc_std'] / ((a + b + c) / 3 + 1e-9)
        f['cell_abc_max'] = max(a, b, c)
        f['cell_abc_min'] = min(a, b, c)
        f['cell_abc_range'] = max(a, b, c) - min(a, b, c)
        f['cell_abc_geo_mean'] = (a * b * c) ** (1/3)
        f['cell_ab_equal'] = float(abs(a - b) < TOL * max(a, b, 1e-9))
        f['cell_ac_equal'] = float(abs(a - c) < TOL * max(a, c, 1e-9))
        f['cell_bc_equal'] = float(abs(b - c) < TOL * max(b, c, 1e-9))
        f['cell_abc_equal'] = float(f['cell_ab_equal'] and f['cell_bc_equal'])
        f['soft_eq_ab'] = 1 - ab_rel
        f['soft_eq_ac'] = 1 - ac_rel
        f['soft_eq_bc'] = 1 - bc_rel
        f['soft_eq_abc'] = f['soft_eq_ab'] * f['soft_eq_ac'] * f['soft_eq_bc']

    if not any(np.isna(v) for v in [al, be, ga]):
        from math import cos, radians, log
        f['cos_alpha'] = cos(radians(al))
        f['cos_beta'] = cos(radians(be))
        f['cos_gamma'] = cos(radians(ga))
        f['angle_dev90_alpha'] = abs(al - 90)
        f['angle_dev90_beta'] = abs(be - 90)
        f['angle_dev90_gamma'] = abs(ga - 90)
        f['angle_dev90_sum'] = abs(al-90) + abs(be-90) + abs(ga-90)
        f['angle_dev90_max'] = max(abs(al-90), abs(be-90), abs(ga-90))
        f['angle_dev120_gamma'] = abs(ga - 120)
        f['angle_all_90'] = float(f['angle_dev90_sum'] < 0.5)
        f['angle_alpha_beta_eq'] = float(abs(al - be) < 0.5)
        f['angle_mean'] = (al + be + ga) / 3
        f['angle_std'] = float(np.std([al, be, ga]))
        dev90 = f['angle_dev90_sum']
        f['score_cubic'] = f.get('soft_eq_abc', 0) / (1 + dev90 + 1e-9)
        f['score_hexagonal'] = f.get('soft_eq_ab', 0) / (1 + abs(ga-120) + abs(al-90) + abs(be-90) + 1e-9)
        f['score_tetragonal'] = f.get('soft_eq_ab', 0) / (1 + dev90 + 1e-9)
        f['score_orthorhombic'] = 1.0 / (1 + dev90 + 1e-9)
        f['score_monoclinic'] = f['angle_alpha_beta_eq'] / (1 + abs(be-90) + 1e-9)

    if not np.isna(V) and V > 1e-9:
        f['log_cell_volume'] = log(V)
        f['cbrt_cell_volume'] = V ** (1/3)

    sg_num = safe_float(find_val(block, '_parent_space_group.IT_number'))
    f['parent_sg_number'] = sg_num
    f['parent_crystal_system'] = get_crystal_system(sg_num)
    if not np.isna(sg_num):
        sg_n = int(sg_num)
        f['parent_sg_normalized'] = sg_n / 230.0
        f['parent_sg_centrosymmetric'] = sg_is_centrosymmetric(sg_n, CENTRO_SET)
        f['parent_pg_order'] = sg_point_group_order(sg_n)
        f['parent_sg_log'] = log(max(sg_n, 1))
        cs = get_crystal_system(sg_n)
        for i in range(1, 8):
            f[f'cs_is_{i}'] = int(cs == i)
        f['cs_low_sym'] = int(cs in [1, 2])
        f['cs_mid_sym'] = int(cs in [3, 4])
        f['cs_high_sym'] = int(cs in [5, 6, 7])
        f['cs_has_hex'] = int(cs in [5, 6])
        f['cs_is_cubic'] = int(cs == 7)
        f['cs_x_soft_eq_ab'] = cs * f.get('soft_eq_ab', 0)
        f['cs_x_soft_eq_abc'] = cs * f.get('soft_eq_abc', 0)
        f['cs_x_angle_dev90'] = cs * f.get('angle_dev90_sum', 0)
        f['cs_x_score_cubic'] = cs * f.get('score_cubic', 0)
        f['cs_x_score_hex'] = cs * f.get('score_hexagonal', 0)
        f['cs_x_pg_order'] = cs * f.get('parent_pg_order', 0)

    elements_all = []
    occupancies = []
    type_col = block.find_loop('_atom_site_type_symbol')
    label_col = block.find_loop('_atom_site_label')
    if col_valid(type_col):
        for s in iter_col_str(type_col):
            sym = elem_from_type_symbol(s)
            if is_valid_elem(sym, INVALID_ELEM):
                elements_all.append(sym)
    elif col_valid(label_col):
        for s in iter_col_str(label_col):
            sym = elem_from_label(s)
            if is_valid_elem(sym, INVALID_ELEM):
                elements_all.append(sym)
    occ_col = block.find_loop('_atom_site_occupancy')
    if col_valid(occ_col):
        for s in iter_col_str(occ_col):
            v = safe_float(s)
            if not np.isna(v):
                occupancies.append(v)
    elements_unique = sorted(set(elements_all))
    n_atoms = len(elements_all)
    n_unique = len(elements_unique)
    f['num_atoms'] = n_atoms
    f['num_unique_elements'] = n_unique
    f['atoms_per_element'] = n_atoms / max(1, n_unique)
    f['composition_complexity'] = n_unique / max(1, n_atoms)

    if occupancies:
        f['occupancy_mean'] = float(np.mean(occupancies))
        f['occupancy_min'] = float(np.min(occupancies))
        f['occupancy_std'] = float(np.std(occupancies))
        f['has_partial_occ'] = int(any(o < 0.99 for o in occupancies))
        f['frac_partial_sites'] = sum(1 for o in occupancies if o < 0.99) / max(1, len(occupancies))
    else:
        f['occupancy_mean'] = 1.0
        f['occupancy_min'] = 1.0
        f['occupancy_std'] = 0.0
        f['has_partial_occ'] = 0
        f['frac_partial_sites'] = 0.0

    fx_col = block.find_loop('_atom_site_fract_x')
    fy_col = block.find_loop('_atom_site_fract_y')
    fz_col = block.find_loop('_atom_site_fract_z')
    if col_valid(fx_col) and col_valid(fy_col) and col_valid(fz_col):
        fxs = [safe_float(s) for s in iter_col_str(fx_col)]
        fys = [safe_float(s) for s in iter_col_str(fy_col)]
        fzs = [safe_float(s) for s in iter_col_str(fz_col)]
        fxs = [v for v in fxs if not np.isna(v)]
        fys = [v for v in fys if not np.isna(v)]
        fzs = [v for v in fzs if not np.isna(v)]
        if fxs and fys and fzs:
            fxa = np.array(fxs)
            fya = np.array(fys)
            fza = np.array(fzs)
            f['frac_x_mean'] = float(np.mean(fxa))
            f['frac_y_mean'] = float(np.mean(fya))
            f['frac_z_mean'] = float(np.mean(fza))
            f['frac_x_std'] = float(np.std(fxa))
            f['frac_y_std'] = float(np.std(fya))
            f['frac_z_std'] = float(np.std(fza))

            def frac_special(arr):
                specials = [0.0, 0.25, 0.5, 0.75, 1.0, 1/3, 2/3, 1/6, 5/6]
                cnt = sum(1 for v in arr if any(abs(v - s) < TOL for s in specials))
                return cnt / max(1, len(arr))
            f['frac_x_special_ratio'] = frac_special(fxa)
            f['frac_y_special_ratio'] = frac_special(fya)
            f['frac_z_special_ratio'] = frac_special(fza)
            f['frac_xyz_special_mean'] = (f['frac_x_special_ratio'] + f['frac_y_special_ratio'] + f['frac_z_special_ratio']) / 3

    el_props = get_elem_props(elements_all, cfg, mendeleev_flag)
    f.update(el_props)
    total_mass = el_props.get('total_mass', 0)
    if not np.isna(V) and V > 1e-9 and total_mass > 0:
        f['density'] = total_mass / V
        f['log_density'] = np.log(total_mass / V)
        f['density_per_atom'] = total_mass / V / max(1, n_atoms)
    if n_atoms > 0 and not np.isna(V):
        f['volume_per_atom'] = V / n_atoms
        f['volume_per_element'] = V / max(1, n_unique)
        f['mass_per_atom'] = total_mass / max(1, n_atoms)
    n_mag = el_props.get('n_magnetic_elements', 0)
    n_re = el_props.get('n_rare_earth_elements', 0)
    n_soc = el_props.get('n_strong_soc_elements', 0)
    cs = f.get('parent_crystal_system', 0) or 0
    dens = f.get('density', 0) or 0

    f['n_mag_x_cs'] = n_mag * cs
    f['n_mag_x_score_cub'] = n_mag * (f.get('score_cubic', 0) or 0)
    f['n_mag_x_score_hex'] = n_mag * (f.get('score_hexagonal', 0) or 0)
    f['n_mag_x_pg_order'] = n_mag * (f.get('parent_pg_order', 0) or 0)
    f['n_re_x_cs'] = n_re * cs
    f['n_soc_x_cs'] = n_soc * cs
    if not np.isna(V) and V > 1e-9:
        f['n_mag_per_volume'] = n_mag / V
        f['n_re_per_volume'] = n_re / V
    f['density_x_cs'] = dens * cs
    f['density_x_pg_order'] = dens * (f.get('parent_pg_order', 0) or 0)
    f['density_x_n_mag'] = dens * n_mag
    f['n_unique_x_cs'] = n_unique * cs
    f['n_atoms_x_cs'] = n_atoms * cs
    f['sym_score_composite'] = f.get('parent_sg_centrosymmetric', 0) * f.get('parent_pg_order', 1) / max(1, n_unique)
    f['mag_density_score'] = n_mag / max(1, n_atoms) * cs
    return f
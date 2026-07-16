import re
import numpy as np
from math import cos, radians, sqrt, log
import warnings
warnings.filterwarnings("ignore")

_ELEM_CACHE = {}

def col_valid(col):
    try:
        return col is not None and bool(col) and len(col) > 0
    except Exception:
        return False

def iter_col_str(col):
    for item in col:
        yield str(item)

def safe_float(val, default=np.nan):
    if val is None:
        return default
    s = str(val).strip()
    if s in ('?', '.', '', 'nan', 'None'):
        return default
    try:
        return float(re.sub(r'\(\d+\)$', '', s))
    except (ValueError, TypeError):
        return default

def find_val(block, tag, default=None):
    try:
        v = block.find_value(tag)
        if v is None:
            return default
        s = str(v).strip()
        return s if s not in ('?', '.', '') else default
    except Exception:
        return default

def elem_from_type_symbol(s):
    s = re.sub(r'[^A-Za-z]', '', str(s).strip())
    if not s:
        return ''
    return s[0].upper() + s[1:].lower()

def elem_from_label(s):
    m = re.match(r'^([A-Za-z]{1,2})', str(s).strip())
    if not m:
        return ''
    raw = m.group(1)
    return raw[0].upper() + raw[1:].lower()

def is_valid_elem(sym, invalid_set):
    if not sym or not (1 <= len(sym) <= 2):
        return False
    if sym in invalid_set or not sym[0].isupper():
        return False
    return True

def cell_volume(a, b, c, al, be, ga):
    if any(np.isnan(v) for v in [a, b, c, al, be, ga]):
        return np.nan
    al_r, be_r, ga_r = radians(al), radians(be), radians(ga)
    cos_term = (1
                - cos(al_r)**2 - cos(be_r)**2 - cos(ga_r)**2
                + 2 * cos(al_r) * cos(be_r) * cos(ga_r))
    cos_term = max(cos_term, 0.0)
    return a * b * c * sqrt(cos_term)

def get_crystal_system(sg_number):
    if np.isna(sg_number):
        return 0
    n = int(sg_number)
    if   1 <= n <=   2: return 1   # triclinic
    elif 3 <= n <=  15: return 2   # monoclinic
    elif 16 <= n <=  74: return 3   # orthorhombic
    elif 75 <= n <= 142: return 4   # tetragonal
    elif 143 <= n <= 167: return 5  # trigonal
    elif 168 <= n <= 194: return 6  # hexagonal
    elif 195 <= n <= 230: return 7  # cubic
    return 0

def sg_is_centrosymmetric(sg_n, centro_set):
    return int(sg_n in centro_set)

def sg_point_group_order(sg_number):
    cs = get_crystal_system(sg_number)
    return {1: 2, 2: 4, 3: 8, 4: 16, 5: 12, 6: 24, 7: 48}.get(cs, 1)

def get_elem_props(symbols_list, cfg, mendeleev_installed):
    if not symbols_list or not mendeleev_installed:
        return {}
    from mendeleev import element as mendeleev_element
    MAGNETIC_ELEMS = set(cfg["magnetic_elems"])
    RARE_EARTH_ELEMS = set(cfg["rare_earth_elems"])
    STRONG_SOC_ELEMS = set(cfg["strong_soc_elems"])

    SCALAR_ATTRS = [
        'atomic_weight', 'en_pauling', 'atomic_radius',
        'electron_affinity', 'vdw_radius', 'dipole_polarizability',
    ]
    collect = {k: [] for k in SCALAR_ATTRS}
    collect['ionization_e1'] = []
    collect['group_id'] = []
    collect['period'] = []

    total_mass = 0.0
    n_mag = n_re = n_tm = n_soc = 0
    atomic_numbers = []

    for sym in symbols_list:
        sym_c = re.sub(r'[^A-Za-z]', '', str(sym)).strip()
        if not sym_c:
            continue
        if sym_c not in _ELEM_CACHE:
            try:
                _ELEM_CACHE[sym_c] = mendeleev_element(sym_c)
            except Exception:
                _ELEM_CACHE[sym_c] = None
        el = _ELEM_CACHE.get(sym_c)
        if el is None:
            continue

        w = el.atomic_weight or 0.0
        total_mass += w
        atomic_numbers.append(el.atomic_number)

        for attr in SCALAR_ATTRS:
            v = getattr(el, attr, None)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                collect[attr].append(float(v))

        try:
            ie1 = el.ionenergies.get(1, None)
            if ie1 is not None:
                collect['ionization_e1'].append(float(ie1))
        except Exception:
            pass

        if el.group_id:
            collect['group_id'].append(int(el.group_id))
        if el.period:
            collect['period'].append(int(el.period))

        if sym_c in MAGNETIC_ELEMS:
            n_mag += 1
        if sym_c in RARE_EARTH_ELEMS:
            n_re += 1
        if el.group_id and 3 <= el.group_id <= 12:
            n_tm += 1
        if sym_c in STRONG_SOC_ELEMS:
            n_soc += 1

    n_total = max(1, len(symbols_list))
    result = {
        'total_mass': total_mass,
        'n_magnetic_elements': n_mag,
        'n_rare_earth_elements': n_re,
        'n_transition_metal_elements': n_tm,
        'n_strong_soc_elements': n_soc,
        'frac_magnetic_elements': n_mag / n_total,
        'frac_rare_earth_elements': n_re / n_total,
        'frac_transition_metal': n_tm / n_total,
        'frac_strong_soc': n_soc / n_total,
        'has_rare_earth_elem': int(n_re > 0),
        'has_magnetic_elem': int(n_mag > 0),
        'has_strong_soc_elem': int(n_soc > 0),
        'has_tm_and_re': int(n_mag > 0 and n_re > 0),
    }

    if atomic_numbers:
        result['atomic_number_mean'] = float(np.mean(atomic_numbers))
        result['atomic_number_max'] = float(np.max(atomic_numbers))
        result['atomic_number_min'] = float(np.min(atomic_numbers))
        result['atomic_number_range'] = float(np.max(atomic_numbers) - np.min(atomic_numbers))

    for col_name, vals in collect.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        result[f'{col_name}_mean'] = float(np.mean(arr))
        result[f'{col_name}_std'] = float(np.std(arr))
        result[f'{col_name}_max'] = float(np.max(arr))
        result[f'{col_name}_min'] = float(np.min(arr))
        result[f'{col_name}_range'] = float(np.max(arr) - np.min(arr))
    return result
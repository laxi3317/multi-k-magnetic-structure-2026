import re
from .cif_utils import find_val

def parse_k_label(block):
    details = find_val(block, '_active_magnetic_irreps_details')
    if not details:
        return 'Other'
    dl = str(details).lower()
    target_k_list = ['4k', '3k', '2k', '1k']
    for kv in target_k_list:
        if re.search(r'\b' + kv + r'\s+magnetic\s+structure\b', dl):
            return kv
    lines = [ln.strip() for ln in dl.split('\n') if ln.strip()]
    if lines:
        for kv in target_k_list:
            if re.search(r'\b' + kv + r'\b', lines[0]):
                return kv
    for kv in target_k_list:
        if kv in dl:
            return kv
    return 'Other'
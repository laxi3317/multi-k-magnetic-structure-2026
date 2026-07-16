import yaml
import numpy as np
from pathlib import Path

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["SEED"]
RANDOM_STATE = CFG["RANDOM_STATE"]
TEST_SIZE = CFG["TEST_SIZE"]
N_SPLITS = CFG["N_SPLITS"]
INPUT_CSV = CFG["INPUT_CSV"]
SAVE_DIR = Path(CFG["SAVE_DIR"])
label3_map = CFG["label3_map"]
EXCLUDE_COLS = CFG["EXCLUDE_COLS"]
BLACKLIST_EXACT = set(CFG["BLACKLIST_EXACT"])
BLACKLIST_SUBSTR = CFG["BLACKLIST_SUBSTR"]
ZERO_FILL_PATTERNS = CFG["ZERO_FILL_PATTERNS"]
S1_TH_INIT = CFG["S1_TH_INIT"]
SMOTE_TARGET = CFG["SMOTE_TARGET"]
KNN_IMP_N = CFG["KNN_IMP_N"]
VAR_THRESH = CFG["VAR_THRESH"]
ET_SEL_EST = CFG["ET_SEL_EST"]

S1_TH_GRID = np.arange(CFG["S1_TH_GRID_START"], CFG["S1_TH_GRID_END"], CFG["S1_TH_GRID_STEP"])
TH_34K_GRID = np.arange(CFG["TH_34K_GRID_START"], CFG["TH_34K_GRID_END"], CFG["TH_34K_GRID_STEP"])
UNCERTAINTY_GRID = CFG["UNCERTAINTY_GRID"]

lgb_s1_params = CFG["lgb_s1_params"]
rf_s1_params = CFG["rf_s1_params"]
lgb_s2_params = CFG["lgb_s2_params"]

FIG_SIZE = CFG["fig_size"]
DPI = CFG["dpi"]

np.random.seed(SEED)
SAVE_DIR.mkdir(exist_ok=True)
import yaml
import os
import numpy as np

with open("config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["seed"]
RANDOM_STATE = CFG["random_state"]
TEST_SIZE = CFG["test_size"]
CV_SPLITS = CFG["cv_splits"]

INPUT_CSV = CFG["input_csv"]
OUTPUT_DIR = CFG["output_dir"]
MODEL_SAVE_DIR = CFG["model_save_dir"]

KNN_IMP_N = CFG["knn_imp_n_neighbors"]
VT_THRESHOLD = CFG["variance_threshold"]
ET_SEL_EST = CFG["et_estimators"]
IMP_PERCENT_THRESH = CFG["imp_percent_cut"]

SMOTE_TARGET = CFG["smote_target_count"]

RECALL_FLOOR = CFG["recall_floor"]
PRECISION_FLOOR = CFG["precision_floor"]
TH_START = CFG["th_start"]
TH_END = CFG["th_end"]
TH_STEP = CFG["th_step"]

LGB_PARAMS = CFG["lgb_params"]
RF_ET_PARAMS = CFG["rf_et_params"]
XGB_PARAMS = CFG["xgb_params"]
LR_C_LIST = CFG["stack_lr"]["c_list"]
LR_MAX_ITER = CFG["stack_lr"]["max_iter"]

EXCLUDE_COLS = CFG["exclude_cols"]
ZERO_FILL_PATTERNS = CFG["zero_fill_patterns"]

COLORS = CFG["colors"]

np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
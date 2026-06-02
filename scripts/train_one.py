import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from configs.reg_base import CFG
from configs.project import CROP_BOX, TRAIN_CSV, VAL_CSV
from printer_ml.train_reg_low_tf import train_regression_low_tf

# Run 1 experiment (change CFG in configs/reg_base.py)
train_regression_low_tf(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    crop_box=CROP_BOX,
    cfg=CFG
)
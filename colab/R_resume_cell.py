# ===== CELL R · Resume after a lost runtime: no download needed (~10 s) =====
# Colab's local disk (/content/gait_raw, /content/gait_imu) is wiped when the runtime resets; Drive is not.
# The confirm cells read only the small result files on Drive (folder w400), so the raw data is not needed.
# Order after a reset: CELL 1 -> CELL R -> 16a -> 16b -> 16c -> 16e (never "Run all": it starts the download).
import json
import matplotlib.pyplot as plt

DATA = RAW_ROOT / "Dataset"                       # raw-data path; only used if the split file were missing
N_STRIDES_CELL14 = 11957                          # stride count printed by CELL 14 on the real data (same for every window)
WINDOW_DIR = "w400"                               # "w400" = primary [-400, -100] ms run (freeze 2026-10-09); "" = old [-600, -100] run
if WINDOW_DIR and OUT_ROOT.name != WINDOW_DIR:
    OUT_ROOT = OUT_ROOT / WINDOW_DIR
log(f"working folder: {OUT_ROOT}")
for f in ["stride_features.parquet", "g1_trials.csv", "dev_confirm_split_v2.json"]:      # inputs of CELL 15
    p = OUT_ROOT / f
    check(p.exists(), f"{f} found on Drive" + (f" ({p.stat().st_size / 1e6:.2f} MB)" if p.exists() else " - MISSING"))
if (OUT_ROOT / "stride_features.parquet").exists():
    n_rows = len(pd.read_parquet(OUT_ROOT / "stride_features.parquet", columns=["subject"]))
    check(n_rows == N_STRIDES_CELL14, f"stride table has {n_rows} strides (CELL 14 gave {N_STRIDES_CELL14})")
if (OUT_ROOT / "dev_confirm_split_v2.json").exists():
    sp_ = json.loads((OUT_ROOT / "dev_confirm_split_v2.json").read_text())
    check(len(sp_["dev"]) == 14 and len(sp_["confirm"]) == 33, f"frozen split v2: {len(sp_['dev'])} dev / {len(sp_['confirm'])} confirm subjects")
end_cell("CELL R")

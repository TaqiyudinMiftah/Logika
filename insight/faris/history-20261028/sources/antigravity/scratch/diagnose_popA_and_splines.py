import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "4"

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit

OUT_ROOT = Path("/mnt/home/faris/logika-joint-20261028")
FRZ_ROOT = Path("/mnt/home/faris/logika-frozen-shadow-20261026")
EXP_ROOT = Path("/mnt/home/faris/logika-faris-exploration")
PERIODIC = EXP_ROOT / "shopping-periodicity-fe-20261021"

sys.path.insert(0, str(EXP_ROOT / "transfer-20261025/source"))
import baseline as b_mod
import feature_head

with open(FRZ_ROOT / "frozen/fold0/MANIFEST.json") as f:
    HEAD18_COLS = json.load(f)["head_columns"]
NAMES = ["power_7d", "power_14d", "power_30d", "periodicity_strength", "dominant_cycle_days"]
EXTRA_PER = ["beh_spectral_" + channel + "_" + name for channel in ["spend", "orders"] for name in NAMES] + ["beh_spectral_window_supported"]
HEAD_PER_COLS = HEAD18_COLS + EXTRA_PER

def ks_stat(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    pos = np.cumsum(ys == 1) / max(np.sum(ys == 1), 1)
    neg = np.cumsum(ys == 0) / max(np.sum(ys == 0), 1)
    diff = np.abs(pos - neg)
    end = np.r_[ps[:-1] != ps[1:], True]
    return float(np.max(diff[end]))

rob_npz = np.load(EXP_ROOT / "transfer-20261025/ROBUSTNESS_OOF.npz", allow_pickle=True)
y_global = rob_npz["target"].astype(np.int8)
per_orig_oof = rob_npz["periodicity__original"].astype(np.float64)
h016_oof = np.load(EXP_ROOT / "transfer-20261025/spline_direct_ks/OOF.npz", allow_pickle=True)["two_seed_mean"].astype(np.float64)

raw_npz = np.load(OUT_ROOT / "raw_domain_features.npz", allow_pickle=True)
raw_names = [str(x) for x in raw_npz["names"]]
raw_data = raw_npz["data"]
pop_masks = {p: raw_data[:, raw_names.index(p)] > 0.5 for p in ["pop_A", "pop_B", "pop_C1", "pop_C2"]}

orig_train_ids = pd.read_parquet(EXP_ROOT / "ranking-pipeline-20261020/input/data/train_features.parquet", columns=["application_id"])["application_id"]
spectral_df = pd.read_parquet(PERIODIC / "true_features.parquet", columns=["application_id"] + EXTRA_PER).set_index("application_id").loc[orig_train_ids].reset_index(drop=True)

roles_list = []
raw_base_70k = np.zeros(70000)
c18_base_70k = np.zeros(70000)
c29_base_70k = np.zeros(70000)
sp_mat_70k = np.zeros(70000)
sp_per_70k = np.zeros(70000)
sp_h16_70k = np.zeros(70000)
sp200_70k = np.load(OUT_ROOT / "linear56k_runs.npz")["val_spline200"]
cb_full_70k = np.load(EXP_ROOT / "catboost-convergence-20261021/OOF.npz")["full"]
cb_comp_70k = np.load(EXP_ROOT / "catboost-convergence-20261021/OOF.npz")["compact"]
lgb56_70k = np.zeros(70000)
cat56_70k = np.zeros(70000)

for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    b = np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz", allow_pickle=True)
    tb = pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet")
    for c in EXTRA_PER:
        tb[c] = spectral_df[c].to_numpy(dtype=np.float32)
    q_idx = np.r_[r["correction"], r["validation"]]
    tb_q = tb.iloc[q_idx].reset_index(drop=True)

    raw_base_70k[r["validation"]] = b["raw"][6000:]
    sp_mat_70k[r["validation"]] = b["spline"]

    h18 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), b["raw"][:6000], HEAD18_COLS, 0.003, 0.5)
    c18_val = b_mod.corrected_probability(h18, tb_q.iloc[6000:], b["raw"][6000:])[0]
    c18_base_70k[r["validation"]] = c18_val

    h29 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), b["raw"][:6000], HEAD_PER_COLS, 0.003, 0.5)
    c29_val = b_mod.corrected_probability(h29, tb_q.iloc[6000:], b["raw"][6000:])[0]
    c29_base_70k[r["validation"]] = c29_val

    sp_per_f = (per_orig_oof[r["validation"]] - 0.75 * c29_val) / 0.25
    sp_per_70k[r["validation"]] = sp_per_f
    h16_val = np.load(EXP_ROOT / f"transfer-20261025/spline_direct_ks/fold{f}/validation.npz")
    sp_h16_70k[r["validation"]] = sp_per_f + h16_val["spline_delta"]

    cf = np.load(OUT_ROOT / f"cpu_runs/fold{f}_crossfit56k.npz")
    lgb56_70k[r["validation"]] = cf["val_full_lgbm_56k"]
    cat56_70k[r["validation"]] = cf["val_full_cat_56k"]

print("=== STAGE-BY-STAGE SUBGROUP KS ANATOMY ===", flush=True)
for name, arr in [
    ("1. Raw Kumo (base_raw)", raw_base_70k),
    ("2. After Head18 (c18_only)", c18_base_70k),
    ("3. After Head29 (c29_only)", c29_base_70k),
    ("4. Maturity Spline77 (sp_mat)", sp_mat_70k),
    ("5. Periodicity Spline88 (sp_per)", sp_per_70k),
    ("6. H016 DirectKS Spline (sp_h16)", sp_h16_70k),
    ("7. Upgraded Spline200 (sp200)", sp200_70k),
    ("8. CatBoost56 Full (cb_full)", cb_full_70k),
    ("9. CatBoost56 Compact (cb_comp)", cb_comp_70k),
    ("10. LightGBM56 Full (lgb56)", lgb56_70k),
    ("11. Baseline Final (75% c18+25% sp_mat)", rob_npz["maturity__original"]),
    ("12. Periodicity Final (75% c29+25% sp_per)", per_orig_oof),
    ("13. H016 Final (2-seed)", h016_oof),
]:
    subs = [f"{pn}:{ks_stat(y_global[pop_masks[pn]], arr[pop_masks[pn]]):.5f}" for pn in ["pop_A", "pop_B", "pop_C1", "pop_C2"]]
    print(f"{name:42s} | PoolKS={ks_stat(y_global, arr):.6f} | " + " | ".join(subs), flush=True)

import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

def ks(y, p):
    f, t, _ = roc_curve(y, p)
    return float(np.max(np.abs(t - f)))

def mean_fold_ks(y, p, folds):
    return float(np.mean([ks(y[folds == f], p[folds == f]) for f in range(5)]))

# Load ground truth and baseline
old = np.load("/mnt/home/faris/logika-faris-exploration/transfer-20261025/ROBUSTNESS_OOF.npz")
y_val = old["target"]
fold_val = old["fold"]
base_oof = old["maturity__original"]

# Load fold roles and baseline components
roles_list = []
base_list = []
y_corr_all = np.zeros(30000, dtype=int)
fold_corr_all = np.zeros(30000, dtype=int)
raw_kumo_corr = np.zeros(30000)
raw_kumo_val = np.zeros(70000)
spline_val = np.zeros(70000)
corr_kumo_val = np.zeros(70000)

for f in range(5):
    r = np.load(f"/mnt/home/faris/logika-frozen-shadow-20261026/frozen/fold{f}/roles.npz")
    b = np.load(f"/mnt/home/faris/logika-frozen-shadow-20261026/frozen/fold{f}/baseline.npz")
    roles_list.append(r)
    base_list.append(b)
    c_slice = slice(f * 6000, (f + 1) * 6000)
    y_corr_all[c_slice] = r["y_correction"]
    fold_corr_all[c_slice] = f
    raw_kumo_corr[c_slice] = b["raw"][:6000]
    vi = r["validation"]
    raw_kumo_val[vi] = b["raw"][6000:]
    spline_val[vi] = b["spline"]
    corr_kumo_val[vi] = b["corrected"]

print("=== BASELINE ANATOMY ===")
for name, p in [
    ("raw_kumo_val", raw_kumo_val),
    ("spline77_val", spline_val),
    ("0.75*raw_kumo + 0.25*spline77", 0.75 * raw_kumo_val + 0.25 * spline_val),
    ("baseline_maturity_original (0.75*corr + 0.25*spline)", base_oof),
]:
    print(f"{name:52s} | Pooled KS: {ks(y_val, p):.6f} | MeanFold KS: {mean_fold_ks(y_val, p, fold_val):.6f} | Gap: {mean_fold_ks(y_val, p, fold_val) - ks(y_val, p):.6f} | AUC: {roc_auc_score(y_val, p):.6f}")

# Check per-fold optimal KS threshold for baseline_maturity_original and raw_kumo_val
print("\n=== PER-FOLD THRESHOLD & CALIBRATION ANATOMY ===")
for f in range(5):
    m = (fold_val == f)
    fpr, tpr, thr = roc_curve(y_val[m], base_oof[m])
    idx = np.argmax(np.abs(tpr - fpr))
    print(f"Fold {f}: Val KS={ks(y_val[m], base_oof[m]):.6f} at thr={thr[idx]:.5f}, tpr={tpr[idx]:.4f}, fpr={fpr[idx]:.4f}, y_mean={y_val[m].mean():.5f}, p_mean={base_oof[m].mean():.5f} | Corr y_mean={roles_list[f]['y_correction'].mean():.5f}, raw_corr_mean={raw_kumo_corr[f*6000:(f+1)*6000].mean():.5f}")

# Collect all 20000-row (6k corr + 14k val) models across campaigns
models = {}
models["base_kumo_champion90"] = (raw_kumo_corr, raw_kumo_val)

def add_campaign_runs(prefix, base_dir, pattern="runs/*"):
    for run_dir in sorted(glob.glob(f"{base_dir}/{pattern}")):
        run_name = run_dir[len(base_dir)+1:].replace("runs/", "")
        if not all(Path(f"{run_dir}/fold{f}/prediction.npz").exists() for f in range(5)):
            continue
        p_corr = np.zeros(30000)
        p_val = np.zeros(70000)
        ok = True
        for f in range(5):
            z = np.load(f"{run_dir}/fold{f}/prediction.npz")
            raw = z["raw"]
            if len(raw) != 20000 or not np.isfinite(raw).all():
                ok = False
                break
            p_corr[f*6000:(f+1)*6000] = raw[:6000]
            p_val[roles_list[f]["validation"]] = raw[6000:]
        if ok:
            models[f"{prefix}:{run_name}"] = (p_corr, p_val)

add_campaign_runs("fs", "/mnt/home/faris/logika-frozen-shadow-20261026", "runs/*")
add_campaign_runs("arch", "/mnt/home/faris/logika-architecture-20261027", "runs/*")
add_campaign_runs("aug90", "/mnt/home/faris/logika-augment90-20261026", "runs/*/*")

# Also check kumo-supervised-view-20261021 and transfer-20261025
if all(Path(f"/mnt/home/faris/logika-faris-exploration/kumo-supervised-view-20261021/folds/fold{f}/raw.npz").exists() for f in range(5)):
    z0 = np.load("/mnt/home/faris/logika-faris-exploration/kumo-supervised-view-20261021/folds/fold0/raw.npz")
    for k in z0.keys():
        if z0[k].dtype.kind not in "fc":
            continue
        p_corr = np.zeros(30000)
        p_val = np.zeros(70000)
        ok = True
        for f in range(5):
            zf = np.load(f"/mnt/home/faris/logika-faris-exploration/kumo-supervised-view-20261021/folds/fold{f}/raw.npz")
            arr = zf[k]
            if len(arr) == 20000:
                p_corr[f*6000:(f+1)*6000] = arr[:6000]
                p_val[roles_list[f]["validation"]] = arr[6000:]
            else:
                ok = False
        if ok:
            models[f"supview:{k}"] = (p_corr, p_val)

# Check transfer-20261025 core_maturity and core_late30
for sub in ["core_maturity", "core_late30"]:
    for kind in ["fresh_raw", "original_raw"]:
        if all(Path(f"/mnt/home/faris/logika-faris-exploration/transfer-20261025/{sub}/fold{f}/{kind}.npz").exists() for f in range(5)):
            z0 = np.load(f"/mnt/home/faris/logika-faris-exploration/transfer-20261025/{sub}/fold0/{kind}.npz")
            for k in z0.keys():
                if z0[k].shape == (20000,) and z0[k].dtype.kind in "fc":
                    p_corr = np.zeros(30000)
                    p_val = np.zeros(70000)
                    for f in range(5):
                        zf = np.load(f"/mnt/home/faris/logika-faris-exploration/transfer-20261025/{sub}/fold{f}/{kind}.npz")
                        p_corr[f*6000:(f+1)*6000] = zf[k][:6000]
                        p_val[roles_list[f]["validation"]] = zf[k][6000:]
                    models[f"transfer:{sub}:{kind}:{k}"] = (p_corr, p_val)

print(f"\nTotal 20k-prediction (6k corr + 14k val) models loaded: {len(models)}")
rows = []
for name, (pc, pv) in models.items():
    rows.append((
        ks(y_corr_all, pc),
        roc_auc_score(y_corr_all, pc),
        ks(y_val, pv),
        mean_fold_ks(y_val, pv, fold_val),
        roc_auc_score(y_val, pv),
        name
    ))

rows.sort(key=lambda x: -x[1]) # sort by Correction AUC!
print("\n=== TOP 35 MODELS BY CORRECTION (30K) AUC ===")
print(f"{'Model':45s} | {'CorrKS':>8s} | {'CorrAUC':>8s} | {'ValKS':>8s} | {'ValMF_KS':>8s} | {'ValAUC':>8s}")
for r in rows[:35]:
    print(f"{r[5]:45s} | {r[0]:8.6f} | {r[1]:8.6f} | {r[2]:8.6f} | {r[3]:8.6f} | {r[4]:8.6f}")

rows_val_ks = sorted(rows, key=lambda x: -x[2])
print("\n=== TOP 25 MODELS BY RAW VALIDATION (70K) KS ===")
for r in rows_val_ks[:25]:
    print(f"{r[5]:45s} | {'CorrKS':>8s}={r[0]:8.6f} | {'CorrAUC':>8s}={r[1]:8.6f} | {'ValKS':>8s}={r[2]:8.6f} | {'ValMF_KS':>8s}={r[3]:8.6f} | {'ValAUC':>8s}={r[4]:8.6f}")

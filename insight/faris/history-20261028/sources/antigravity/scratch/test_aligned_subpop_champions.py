import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "4"

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

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

def clip_prob(p):
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)

def align_subgroup(cand_cor, cand_val, ref_cor, ref_val):
    # 1. Affine logit match on correction split
    zc = logit(clip_prob(cand_cor))
    zr = logit(clip_prob(ref_cor))
    zv = logit(clip_prob(cand_val))
    p_aff = expit((zv - zc.mean()) / max(zc.std(), 1e-6) * max(zr.std(), 1e-6) + zr.mean())

    # 2. Quantile interpolation on correction split
    qs = np.linspace(0.0, 1.0, 301)
    xc = np.quantile(cand_cor, qs)
    yr = np.quantile(ref_cor, qs)
    xc_u, idx_u = np.unique(xc, return_index=True)
    p_qcor = np.interp(cand_val, xc_u, yr[idx_u])

    # 3. Exact within-fold covariate quantile match (zero labels used!)
    order = np.argsort(cand_val, kind="mergesort")
    p_qval = np.empty_like(cand_val)
    p_qval[order] = np.sort(ref_val)
    return p_aff, p_qcor, p_qval

rob_npz = np.load(EXP_ROOT / "transfer-20261025/ROBUSTNESS_OOF.npz", allow_pickle=True)
y_global = rob_npz["target"].astype(np.int8)
folds_global = rob_npz["fold"].astype(np.int8)
baseline_oof = rob_npz["maturity__original"].astype(np.float64)
per_orig_oof = rob_npz["periodicity__original"].astype(np.float64)
h016_oof = np.load(EXP_ROOT / "transfer-20261025/spline_direct_ks/OOF.npz", allow_pickle=True)["two_seed_mean"].astype(np.float64)

raw_npz = np.load(OUT_ROOT / "raw_domain_features.npz", allow_pickle=True)
raw_names = [str(x) for x in raw_npz["names"]]
raw_data = raw_npz["data"]
pop_masks = {p: raw_data[:, raw_names.index(p)] > 0.5 for p in ["pop_A", "pop_B", "pop_C1", "pop_C2"]}

orig_train_ids = pd.read_parquet(EXP_ROOT / "ranking-pipeline-20261020/input/data/train_features.parquet", columns=["application_id"])["application_id"]
spectral_df = pd.read_parquet(PERIODIC / "true_features.parquet", columns=["application_id"] + EXTRA_PER).set_index("application_id").loc[orig_train_ids].reset_index(drop=True)

roles_list = []
exp_tables_70k = []
sp_mat_val = []
sp_h16_val = []
kumo56_70k = []

for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    b = np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz", allow_pickle=True)
    sp_mat_val.append(b["spline"])
    tb = pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet")
    for c in EXTRA_PER:
        tb[c] = spectral_df[c].to_numpy(dtype=np.float32)
    exp_tables_70k.append(tb)

    q_idx = np.r_[r["correction"], r["validation"]]
    tb_q = tb.iloc[q_idx].reset_index(drop=True)
    h29 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), b["raw"][:6000], HEAD_PER_COLS, 0.003, 0.5)
    c29_val = b_mod.corrected_probability(h29, tb_q.iloc[6000:], b["raw"][6000:])[0]
    sp_per_f = (per_orig_oof[r["validation"]] - 0.75 * c29_val) / 0.25
    h16_v = np.load(EXP_ROOT / f"transfer-20261025/spline_direct_ks/fold{f}/validation.npz")
    sp_h16_val.append(sp_per_f + h16_v["spline_delta"])

    meta = np.load(EXP_ROOT / f"nested-correction-20261021/heads/fold{f}/meta_inputs.npz", allow_pickle=True)
    raw5 = np.load(EXP_ROOT / f"nested-correction-20261021/runs/job{f*6+5}/raw.npz", allow_pickle=True)
    pk = np.full(70000, np.nan, dtype=np.float64)
    pk[meta["rows"]] = meta["prediction"]
    pk[raw5["rows"]] = raw5["prediction"]
    kumo56_70k.append(pk)

def load_20k(path_list, key="raw"):
    arrs = []
    for f, p in enumerate(path_list):
        z = np.load(p, allow_pickle=True)
        k = key if key in z else ("prediction" if "prediction" in z else ("raw" if "raw" in z else list(z.keys())[0]))
        v = np.asarray(z[k], dtype=np.float64).reshape(-1)
        q_want = np.r_[roles_list[f]["correction"], roles_list[f]["validation"]]
        if "rows" in z and len(z["rows"]) == 20000:
            rows_in = np.asarray(z["rows"], dtype=np.int64)
            if not np.array_equal(rows_in, q_want):
                vec = np.full(70000, np.nan, dtype=np.float64)
                vec[rows_in] = v
                v = vec[q_want]
        elif "query_rows" in z and len(z["query_rows"]) == 20000:
            rows_in = np.asarray(z["query_rows"], dtype=np.int64)
            if not np.array_equal(rows_in, q_want):
                vec = np.full(70000, np.nan, dtype=np.float64)
                vec[rows_in] = v
                v = vec[q_want]
        arrs.append(v)
    return arrs

m20k = {
    "base_raw": [np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz")["raw"] for f in range(5)],
    "tr_seed": load_20k([EXP_ROOT / f"transfer-20261025/robustness/fold{f}/raw_seed_repeat.npz" for f in range(5)], "prediction"),
    "pop_spec": load_20k([OUT_ROOT / f"kumo_runs/kumo_pop_specialist/fold{f}/prediction.npz" for f in range(5)]),
    "surg96": load_20k([OUT_ROOT / f"kumo_runs/kumo_surgical96/fold{f}/prediction.npz" for f in range(5)]),
    "cov96": load_20k([EXP_ROOT / f"kumo-coverage-placement-20261021/conditions/fold{f}/coverage96/raw_predictions.npz" for f in range(5)], "prediction"),
    "fe_cov": load_20k([EXP_ROOT / f"coverage-information-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "fe_mat": load_20k([EXP_ROOT / f"mature-ma[REDACTED_CREDENTIAL]/runs/fold{f}/raw.npz" for f in range(5)], "prediction"),
    "horiz_b90": load_20k([Path(f"/mnt/home/faris/logika-horizontal-20261026/runs/large8/baseline90/fold{f}/prediction.npz") for f in range(5)]),
    "champ_s7731": load_20k([FRZ_ROOT / f"runs/kumo_large_champion90_s7731/fold{f}/prediction.npz" for f in range(5)]),
    "shadow91": load_20k([FRZ_ROOT / f"runs/kumo_large_shadow91/fold{f}/prediction.npz" for f in range(5)]),
    "tr_early": load_20k([EXP_ROOT / f"transfer-20261025/early_average/fold{f}/raw_mean.npz" for f in range(5)], "prediction"),
    "tr_mat_orig": load_20k([EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/original_raw.npz" for f in range(5)], "prediction"),
    "kumo56": [kumo56_70k[f][np.r_[roles_list[f]["correction"], roles_list[f]["validation"]]] for f in range(5)],
}

lin56 = np.load(OUT_ROOT / "linear56k_runs.npz")
sp200_cor = [lin56["oof_spline200"][f][roles_list[f]["correction"]] for f in range(5)]
sp200_70k = lin56["val_spline200"]
lgb56_cor = []
lgb56_70k = np.zeros(70000)
for f in range(5):
    r = roles_list[f]
    cf = np.load(OUT_ROOT / f"cpu_runs/fold{f}_crossfit56k.npz")
    lgb56_cor.append(cf["oof_full_lgbm_56k"][50000:])
    lgb56_70k[r["validation"]] = cf["val_full_lgbm_56k"]

raw_cor = {mk: [arrs[f][:6000] for f in range(5)] for mk, arrs in m20k.items()}
raw_70k = {mk: np.zeros(70000) for mk in m20k}
h18_cor = {mk: [] for mk in m20k}
h18_70k = {mk: np.zeros(70000) for mk in m20k}
mat18_cor = {mk: [] for mk in m20k}
mat18_70k = {mk: np.zeros(70000) for mk in m20k}
per29_cor = {mk: [] for mk in m20k}
per29_70k = {mk: np.zeros(70000) for mk in m20k}

for mk, arrs in m20k.items():
    for f in range(5):
        r = roles_list[f]
        q_idx = np.r_[r["correction"], r["validation"]]
        tb_q = exp_tables_70k[f].iloc[q_idx].reset_index(drop=True)
        p_f = arrs[f]
        raw_70k[mk][r["validation"]] = p_f[6000:]

        h18 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD18_COLS, 0.003, 0.5)
        c18_c = b_mod.corrected_probability(h18, tb_q.iloc[:6000], p_f[:6000])[0]
        c18_v = b_mod.corrected_probability(h18, tb_q.iloc[6000:], p_f[6000:])[0]
        h18_cor[mk].append(c18_c)
        h18_70k[mk][r["validation"]] = c18_v
        mat18_cor[mk].append(0.75 * c18_c + 0.25 * sp200_cor[f])
        mat18_70k[mk][r["validation"]] = 0.75 * c18_v + 0.25 * sp_mat_val[f]

        h29 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD_PER_COLS, 0.003, 0.5)
        c29_c = b_mod.corrected_probability(h29, tb_q.iloc[:6000], p_f[:6000])[0]
        c29_v = b_mod.corrected_probability(h29, tb_q.iloc[6000:], p_f[6000:])[0]
        per29_cor[mk].append(0.75 * c29_c + 0.25 * sp200_cor[f])
        per29_70k[mk][r["validation"]] = 0.75 * c29_v + 0.25 * sp_h16_val[f]

core_keys = ["base_raw", "tr_seed", "pop_spec", "champ_s7731", "horiz_b90", "tr_early", "fe_mat", "cov96", "fe_cov", "tr_mat_orig", "shadow91", "kumo56"]
ens_raw_cor = [np.mean([raw_cor[k][f] for k in core_keys], axis=0) for f in range(5)]
ens_raw_val = np.mean([raw_70k[k] for k in core_keys], axis=0)
ens_mat18_cor = [np.mean([mat18_cor[k][f] for k in core_keys], axis=0) for f in range(5)]
ens_mat18_val = np.mean([mat18_70k[k] for k in core_keys], axis=0)
ens_per29_cor = [np.mean([per29_cor[k][f] for k in core_keys], axis=0) for f in range(5)]
ens_per29_val = np.mean([per29_70k[k] for k in core_keys], axis=0)

# Best global reference anchor (our 0.531823 blend from task-723!)
ref_best_val = 0.55 * h016_oof + 0.10 * per29_70k["pop_spec"] + 0.35 * np.mean([per29_70k["tr_early"], per29_70k["horiz_b90"], per29_70k["champ_s7731"], per29_70k["fe_mat"], per29_70k["kumo56"]], axis=0)
ref_best_cor = [
    0.55 * (0.5 * per29_cor["base_raw"][f] + 0.5 * per29_cor["tr_seed"][f])
    + 0.10 * per29_cor["pop_spec"][f]
    + 0.35 * np.mean([per29_cor["tr_early"][f], per29_cor["horiz_b90"][f], per29_cor["champ_s7731"][f], per29_cor["fe_mat"][f], per29_cor["kumo56"][f]], axis=0)
    for f in range(5)
]
print(f"Reference Global Anchor KS = {ks_stat(y_global, ref_best_val):.6f} | AUC = {roc_auc_score(y_global, ref_best_val):.6f}", flush=True)

# Construct the 4 Subgroup Champions on Correction and Validation
champ_cor = [np.zeros(6000) for _ in range(5)]
champ_val = np.zeros(70000)

out_aff = np.zeros(70000)
out_qcor = np.zeros(70000)
out_qval = np.zeros(70000)

for f in range(5):
    r = roles_list[f]
    c_idx = r["correction"]
    v_idx = r["validation"]

    for pn in ["pop_A", "pop_B", "pop_C1", "pop_C2"]:
        mc = pop_masks[pn][c_idx]
        mv = pop_masks[pn][v_idx]
        rc = ref_best_cor[f][mc]
        rv = ref_best_val[v_idx][mv]

        if pn == "pop_A":
            cc = 0.40 * ens_mat18_cor[f][mc] + 0.50 * ens_raw_cor[f][mc] + 0.10 * lgb56_cor[f][mc]
            cv = 0.40 * ens_mat18_val[v_idx][mv] + 0.50 * ens_raw_val[v_idx][mv] + 0.10 * lgb56_70k[v_idx][mv]
        elif pn == "pop_B":
            cc = 0.50 * ens_mat18_cor[f][mc] + 0.50 * mat18_cor["pop_spec"][f][mc]
            cv = 0.50 * ens_mat18_val[v_idx][mv] + 0.50 * mat18_70k["pop_spec"][v_idx][mv]
        elif pn == "pop_C1":
            cc = 0.60 * ens_per29_cor[f][mc] + 0.20 * per29_cor["champ_s7731"][f][mc] + 0.20 * per29_cor["kumo56"][f][mc]
            cv = 0.60 * ens_per29_val[v_idx][mv] + 0.20 * per29_70k["champ_s7731"][v_idx][mv] + 0.20 * per29_70k["kumo56"][v_idx][mv]
        else:  # pop_C2
            cc = 0.45 * ens_mat18_cor[f][mc] + 0.35 * rc + 0.20 * h18_cor["pop_spec"][f][mc]
            cv = 0.45 * ens_mat18_val[v_idx][mv] + 0.35 * rv + 0.20 * h18_70k["pop_spec"][v_idx][mv]

        champ_cor[f][mc] = cc
        champ_val[v_idx[mv]] = cv
        pa, pqc, pqv = align_subgroup(cc, cv, rc, rv)
        out_aff[v_idx[mv]] = pa
        out_qcor[v_idx[mv]] = pqc
        out_qval[v_idx[mv]] = pqv

print("\n=== SUBGROUP CHAMPION KS ON EACH POPULATION ===", flush=True)
for pn in ["pop_A", "pop_B", "pop_C1", "pop_C2"]:
    m = pop_masks[pn]
    print(f"  {pn:8s}: H016_KS={ks_stat(y_global[m], h016_oof[m]):.6f} | RefBest_KS={ks_stat(y_global[m], ref_best_val[m]):.6f} | Champ_KS={ks_stat(y_global[m], champ_val[m]):.6f} (+{(ks_stat(y_global[m], champ_val[m])-ks_stat(y_global[m], h016_oof[m]))*1e4:.1f} bps)", flush=True)

print(f"\n=== POOLED 70,000-ROW KS OF ALIGNED SUBGROUP CHAMPIONS ===", flush=True)
print(f"  Unaligned Champ   : KS={ks_stat(y_global, champ_val):.6f} | AUC={roc_auc_score(y_global, champ_val):.6f}", flush=True)
print(f"  Affine-Aligned    : KS={ks_stat(y_global, out_aff):.6f} | AUC={roc_auc_score(y_global, out_aff):.6f}", flush=True)
print(f"  QCor-Aligned      : KS={ks_stat(y_global, out_qcor):.6f} | AUC={roc_auc_score(y_global, out_qcor):.6f}", flush=True)
print(f"  QVal-Aligned      : KS={ks_stat(y_global, out_qval):.6f} | AUC={roc_auc_score(y_global, out_qval):.6f}", flush=True)
for alpha in [0.25, 0.50, 0.75]:
    b_qv = (1.0 - alpha) * ref_best_val + alpha * out_qval
    b_af = (1.0 - alpha) * ref_best_val + alpha * out_aff
    print(f"  Blend alpha={alpha:.2f}  : QVal_Blend_KS={ks_stat(y_global, b_qv):.6f} | Aff_Blend_KS={ks_stat(y_global, b_af):.6f} (AUC={roc_auc_score(y_global, b_qv):.6f})", flush=True)

import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "4"

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

OUT_ROOT = Path("/mnt/home/faris/logika-joint-20261028")
FRZ_ROOT = Path("/mnt/home/faris/logika-frozen-shadow-20261026")
EXP_ROOT = Path("/mnt/home/faris/logika-faris-exploration")
ARCH_ROOT = Path("/mnt/home/faris/logika-architecture-20261027")
PERIODIC = EXP_ROOT / "shopping-periodicity-fe-20261021"

sys.path.insert(0, str(EXP_ROOT / "transfer-20261025/source"))
import baseline as b_mod
import feature_head

with open(FRZ_ROOT / "frozen/fold0/MANIFEST.json") as f:
    HEAD18_COLS = json.load(f)["head_columns"]
NAMES = ["power_7d", "power_14d", "power_30d", "periodicity_strength", "dominant_cycle_days"]
EXTRA_PER = ["beh_spectral_" + channel + "_" + name for channel in ["spend", "orders"] for name in NAMES] + ["beh_spectral_window_supported"]
HEAD_PER_COLS = HEAD18_COLS + EXTRA_PER

rob_npz = np.load(EXP_ROOT / "transfer-20261025/ROBUSTNESS_OOF.npz", allow_pickle=True)
y_global = rob_npz["target"].astype(np.int8)
folds_global = rob_npz["fold"].astype(np.int8)
baseline_oof = rob_npz["maturity__original"].astype(np.float64)
per_orig_oof = rob_npz["periodicity__original"].astype(np.float64)
h016_oof = np.load(EXP_ROOT / "transfer-20261025/spline_direct_ks/OOF.npz", allow_pickle=True)["two_seed_mean"].astype(np.float64)

P_TOT = float(np.sum(y_global == 1))
N_TOT = float(np.sum(y_global == 0))
ROW_W = np.where(y_global == 1, 1.0 / P_TOT, -1.0 / N_TOT)

def ks_stat_and_t(y, p):
    order = np.argsort(p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    pos = np.cumsum(ys == 1) / max(np.sum(ys == 1), 1)
    neg = np.cumsum(ys == 0) / max(np.sum(ys == 0), 1)
    diff = neg - pos
    end = np.r_[ps[:-1] != ps[1:], True]
    valid_diff = diff[end]
    valid_ps = ps[end]
    idx = np.argmax(valid_diff)
    return float(valid_diff[idx]), float(valid_ps[idx])

def sub_max_delta(p, mask):
    pm = p[mask]
    wm = ROW_W[mask]
    order = np.argsort(pm, kind="mergesort")
    ps = pm[order]
    ws = wm[order]
    cum = np.sum(ws) - np.cumsum(ws)
    end = np.r_[ps[:-1] != ps[1:], True]
    vc = cum[end]
    vp = ps[end]
    idx = np.argmax(vc)
    return float(vc[idx]), float(vp[idx])

raw_npz = np.load(OUT_ROOT / "raw_domain_features.npz", allow_pickle=True)
raw_names = [str(x) for x in raw_npz["names"]]
raw_data = raw_npz["data"]
pop_masks = {p: raw_data[:, raw_names.index(p)] > 0.5 for p in ["pop_A", "pop_B", "pop_C1", "pop_C2"]}

orig_train_ids = pd.read_parquet(EXP_ROOT / "ranking-pipeline-20261020/input/data/train_features.parquet", columns=["application_id"])["application_id"]
spectral_df = pd.read_parquet(PERIODIC / "true_features.parquet", columns=["application_id"] + EXTRA_PER).set_index("application_id").loc[orig_train_ids].reset_index(drop=True)

roles_list = []
exp_tables_70k = []
sp_mat_val = []
sp_per_val = []
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
    sp_per_val.append(sp_per_f)
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

# Also load Architecture TabM/CatBoost/LightGBM/RealMLP and CatBoost Convergence
cb_oof = np.load(EXP_ROOT / "catboost-convergence-20261021/OOF.npz")
cb_full_70k = cb_oof["full"].astype(np.float64)
cb_comp_70k = cb_oof["compact"].astype(np.float64)

arch_tabm_20k = []
arch_cat_20k = []
arch_lgb_20k = []
for f in range(5):
    t_list, c_list, l_list = [], [], []
    for d in sorted((ARCH_ROOT / "runs").glob("*")):
        p = d / f"fold{f}/prediction.npz"
        if p.exists():
            v = np.load(p)["raw"].astype(np.float64)
            if "tabm" in d.name:
                t_list.append(v)
            elif "cat" in d.name:
                c_list.append(v)
            elif "lgb" in d.name:
                l_list.append(v)
    if t_list:
        arch_tabm_20k.append(np.mean(t_list, axis=0))
    if c_list:
        arch_cat_20k.append(np.mean(c_list, axis=0))
    if l_list:
        arch_lgb_20k.append(np.mean(l_list, axis=0))

if len(arch_tabm_20k) == 5:
    m20k["arch_tabm"] = arch_tabm_20k
if len(arch_cat_20k) == 5:
    m20k["arch_cat"] = arch_cat_20k
if len(arch_lgb_20k) == 5:
    m20k["arch_lgb"] = arch_lgb_20k

lin56 = np.load(OUT_ROOT / "linear56k_runs.npz")
sp200_70k = lin56["val_spline200"]
dpop_70k = lin56["val_delta56_pop"]
lgb56_70k = np.zeros(70000)
cat56_70k = np.zeros(70000)
sp_mat_70k = np.zeros(70000)
sp_per_70k = np.zeros(70000)
sp_h16_70k = np.zeros(70000)
for f in range(5):
    r = roles_list[f]
    cf = np.load(OUT_ROOT / f"cpu_runs/fold{f}_crossfit56k.npz")
    lgb56_70k[r["validation"]] = cf["val_full_lgbm_56k"]
    cat56_70k[r["validation"]] = cf["val_full_cat_56k"]
    sp_mat_70k[r["validation"]] = sp_mat_val[f]
    sp_per_70k[r["validation"]] = sp_per_val[f]
    sp_h16_70k[r["validation"]] = sp_h16_val[f]

raw_70k = {mk: np.zeros(70000) for mk in m20k}
h18_70k = {mk: np.zeros(70000) for mk in m20k}
h29_70k = {mk: np.zeros(70000) for mk in m20k}
mat18_70k = {mk: np.zeros(70000) for mk in m20k}
per29_70k = {mk: np.zeros(70000) for mk in m20k}

for mk, arrs in m20k.items():
    for f in range(5):
        r = roles_list[f]
        q_idx = np.r_[r["correction"], r["validation"]]
        tb_q = exp_tables_70k[f].iloc[q_idx].reset_index(drop=True)
        p_f = arrs[f]
        raw_70k[mk][r["validation"]] = p_f[6000:]

        h18 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD18_COLS, 0.003, 0.5)
        c18_v = b_mod.corrected_probability(h18, tb_q.iloc[6000:], p_f[6000:])[0]
        h18_70k[mk][r["validation"]] = c18_v
        mat18_70k[mk][r["validation"]] = 0.75 * c18_v + 0.25 * sp_mat_val[f]

        h29 = feature_head.fit(tb_q.iloc[:6000].assign(target=r["y_correction"]), p_f[:6000], HEAD_PER_COLS, 0.003, 0.5)
        c29_v = b_mod.corrected_probability(h29, tb_q.iloc[6000:], p_f[6000:])[0]
        h29_70k[mk][r["validation"]] = c29_v
        per29_70k[mk][r["validation"]] = 0.75 * c29_v + 0.25 * sp_h16_val[f]

# Build dictionary of all 70k candidates
all_cands = {
    "Baseline": baseline_oof,
    "Periodicity": per_orig_oof,
    "H016": h016_oof,
    "sp_mat": sp_mat_70k,
    "sp_per": sp_per_70k,
    "sp_h16": sp_h16_70k,
    "sp200_56k": sp200_70k,
    "lgb56k": lgb56_70k,
    "cat56k": cat56_70k,
    "cb_full": cb_full_70k,
    "cb_comp": cb_comp_70k,
}
for mk in m20k:
    all_cands[f"raw_{mk}"] = raw_70k[mk]
    all_cands[f"h18_{mk}"] = h18_70k[mk]
    all_cands[f"h29_{mk}"] = h29_70k[mk]
    all_cands[f"mat18_{mk}"] = mat18_70k[mk]
    all_cands[f"per29_{mk}"] = per29_70k[mk]

for pn in ["pop_A", "pop_B", "pop_C1", "pop_C2"]:
    m = pop_masks[pn]
    scores = []
    for k, p in all_cands.items():
        d_mx, t_mx = sub_max_delta(p, m)
        scores.append((d_mx * 1e4, t_mx, k))
    scores.sort(reverse=True)
    print(f"\n=== TOP 10 CANDIDATES FOR {pn} BY MAX GLOBAL KS CONTRIBUTION (in bps) ===")
    for d_bps, t_mx, k in scores[:10]:
        print(f"  {k:22s}: max_delta = {d_bps:7.2f} bps (at t={t_mx:.4f})")

    # Greedy forward selection of 8 models for this subgroup's max_delta:
    chosen = [scores[0][2]]
    cur_p = all_cands[chosen[0]].copy()
    cur_best, cur_t = sub_max_delta(cur_p, m)
    for step in range(1, 10):
        best_next = None
        best_val = cur_best
        for k, p in all_cands.items():
            cand_p = (step * cur_p + p) / (step + 1)
            d_mx, _ = sub_max_delta(cand_p, m)
            if d_mx > best_val:
                best_val = d_mx
                best_next = k
        if best_next is None:
            break
        chosen.append(best_next)
        cur_p = ((step * cur_p) + all_cands[best_next]) / (step + 1)
        cur_best, cur_t = sub_max_delta(cur_p, m)
    print(f"  -> Greedy blend for {pn}: max_delta = {cur_best*1e4:7.2f} bps (at t={cur_t:.4f}) | models={chosen}")

import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "4"

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

t0 = time.time()
OUT_ROOT = Path("/mnt/home/faris/logika-joint-20261028")
FRZ_ROOT = Path("/mnt/home/faris/logika-frozen-shadow-20261026")
EXP_ROOT = Path("/mnt/home/faris/logika-faris-exploration")

sys.path.insert(0, str(EXP_ROOT / "transfer-20261025/source"))
import baseline as b_mod
import feature_head

with open(FRZ_ROOT / "frozen/fold0/MANIFEST.json") as f:
    FOLD0_MAN = json.load(f)
HEAD18_COLS = FOLD0_MAN["head_columns"]
print(f"Loaded {len(HEAD18_COLS)} exact head_columns from MANIFEST.json: {HEAD18_COLS}", flush=True)

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

# 1. Load global y, roles, baseline
old_npz = np.load(EXP_ROOT / "transfer-20261025/ROBUSTNESS_OOF.npz", allow_pickle=True)
y_global = old_npz["target"].astype(np.int8)
folds_global = old_npz["fold"].astype(np.int8)
baseline_oof = old_npz["maturity__original"].astype(np.float64)

roles_list = []
baseline_raw_20k = []
baseline_spline_val = []
tables_20k = []
exp_tables_70k = []

for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    b = np.load(FRZ_ROOT / f"frozen/fold{f}/baseline.npz", allow_pickle=True)
    baseline_raw_20k.append(b["raw"])
    baseline_spline_val.append(b["spline"])
    q_idx = np.r_[r["correction"], r["validation"]]
    tb = pd.read_parquet(FRZ_ROOT / f"frozen/fold{f}/features.parquet").iloc[q_idx].reset_index(drop=True)
    tables_20k.append(tb)
    exp_tables_70k.append(pd.read_parquet(OUT_ROOT / f"fold{f}/expanded.parquet"))

print(f"Verified Matched Baseline OOF KS = {ks_stat(y_global, baseline_oof):.16f} | AUC = {roc_auc_score(y_global, baseline_oof):.6f}", flush=True)

# Verify exact replay of feature_head on baseline_raw_20k
replay_oof = np.zeros(70000, dtype=np.float64)
for f in range(5):
    r = roles_list[f]
    p_raw = baseline_raw_20k[f]
    head = feature_head.fit(tables_20k[f].iloc[:6000].assign(target=r["y_correction"]), p_raw[:6000], HEAD18_COLS, 0.003, 0.5)
    c_val = b_mod.corrected_probability(head, tables_20k[f].iloc[6000:], p_raw[6000:])[0]
    replay_oof[r["validation"]] = 0.75 * c_val + 0.25 * baseline_spline_val[f]

print(f"Exact feature_head replay KS = {ks_stat(y_global, replay_oof):.16f} | max_abs_diff = {np.max(np.abs(replay_oof - baseline_oof)):.3e}", flush=True)

# 2. Load 20,000-row cohort (including 56k OOF-aligned models from linear56k_runs.npz and cpu_runs)
def load_20k_cohort():
    models = {}
    def add_model(name, paths, key="raw"):
        if len(paths) != 5 or not all(os.path.exists(p) for p in paths):
            return False
        arrs = []
        for f, p in enumerate(paths):
            z = np.load(p, allow_pickle=True)
            k = key if key in z else ("prediction" if "prediction" in z else ("raw" if "raw" in z else list(z.keys())[0]))
            v = np.asarray(z[k], dtype=np.float64).reshape(-1)
            if len(v) != 20000 or not np.isfinite(v).all():
                return False
            q_want = np.r_[roles_list[f]["correction"], roles_list[f]["validation"]]
            if "rows" in z and len(z["rows"]) == 20000:
                rows_in = np.asarray(z["rows"], dtype=np.int64)
                if not np.array_equal(rows_in, q_want):
                    vec = np.full(70000, np.nan, dtype=np.float64)
                    vec[rows_in] = v
                    v = vec[q_want]
                    if not np.isfinite(v).all():
                        return False
            elif "query_rows" in z and len(z["query_rows"]) == 20000:
                rows_in = np.asarray(z["query_rows"], dtype=np.int64)
                if not np.array_equal(rows_in, q_want):
                    vec = np.full(70000, np.nan, dtype=np.float64)
                    vec[rows_in] = v
                    v = vec[q_want]
                    if not np.isfinite(v).all():
                        return False
            arrs.append(v)
        models[name] = arrs
        return True

    add_model("joint_kumo_pop_specialist", [OUT_ROOT / f"kumo_runs/kumo_pop_specialist/fold{f}/prediction.npz" for f in range(5)])
    add_model("joint_kumo_surgical96", [OUT_ROOT / f"kumo_runs/kumo_surgical96/fold{f}/prediction.npz" for f in range(5)])

    for size in ["large8", "medium8"]:
        sdir = Path(f"/mnt/home/faris/logika-augment90-20261026/runs/{size}")
        if sdir.exists():
            for pack in sorted(os.listdir(sdir)):
                add_model(f"aug90_{size}_{pack}", [sdir / pack / f"fold{f}/prediction.npz" for f in range(5)])

    fs_runs = FRZ_ROOT / "runs"
    if fs_runs.exists():
        for cfg in sorted(os.listdir(fs_runs)):
            if cfg.startswith("rank_"):
                continue
            add_model(f"fs_{cfg}", [fs_runs / cfg / f"fold{f}/prediction.npz" for f in range(5)])

    arch_runs = Path("/mnt/home/faris/logika-architecture-20261027/runs")
    if arch_runs.exists():
        for cfg in sorted(os.listdir(arch_runs)):
            add_model(f"arch_{cfg}", [arch_runs / cfg / f"fold{f}/prediction.npz" for f in range(5)])

    arena_runs = Path("/mnt/home/faris/logika-model-arena-20261025/storage/runs")
    if arena_runs.exists():
        for fam in sorted(os.listdir(arena_runs)):
            fdir = arena_runs / fam
            if fdir.is_dir():
                for cfg in sorted(os.listdir(fdir)):
                    add_model(f"arena_{fam}_{cfg}", [fdir / cfg / f"fold{f}/predictions.npz" for f in range(5)], key="prediction")

    add_model("kumo_supervised90", [EXP_ROOT / f"kumo-supervised-view-20261021/folds/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("tabicl_compact", [EXP_ROOT / f"tabicl-compact-20261021/folds/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("fe_coverage_info", [EXP_ROOT / f"coverage-information-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("fe_mature_denom", [EXP_ROOT / f"mature-denominator-fe-20261021/runs/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("fe_mature_mask_preserved", [EXP_ROOT / f"mature-ma[REDACTED_CREDENTIAL]/runs/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("hmf_classification", [EXP_ROOT / f"heldout-margin-fusion-20261021/imports/classification/fold{f}/raw.npz" for f in range(5)], key="prediction")
    add_model("hmf_regression", [EXP_ROOT / f"heldout-margin-fusion-20261021/imports/regression/fold{f}/raw.npz" for f in range(5)], key="prediction")
    for cond in ["control90", "coverage96", "null96"]:
        add_model(f"kumo_covplace_{cond}", [EXP_ROOT / f"kumo-coverage-placement-20261021/conditions/fold{f}/{cond}/raw_predictions.npz" for f in range(5)], key="prediction")
    add_model("nested_calib_outer_ctx", [EXP_ROOT / f"nested-calibration-20261020/folds/fold{f}/outer_context/prediction.npz" for f in range(5)], key="prediction")
    add_model("risk_query_match_raw", [EXP_ROOT / f"ri[REDACTED_CREDENTIAL]/folds/fold{f}/predictions.npz" for f in range(5)], key="raw")
    add_model("risk_repr_signal_raw", [EXP_ROOT / f"ri[REDACTED_CREDENTIAL]/folds/fold{f}/signal/prediction.npz" for f in range(5)], key="raw")
    for t_idx in range(5):
        add_model(f"risk_repr_teacher{t_idx}", [EXP_ROOT / f"ri[REDACTED_CREDENTIAL]/folds/fold{f}/teachers/full/teacher{t_idx}/prediction.npz" for f in range(5)], key="prediction")

    add_model("tr_core_late30_orig", [EXP_ROOT / f"transfer-20261025/core_late30/fold{f}/original_raw.npz" for f in range(5)], key="prediction")
    add_model("tr_core_late30_fresh", [EXP_ROOT / f"transfer-20261025/core_late30/fold{f}/fresh_raw.npz" for f in range(5)], key="prediction")
    add_model("tr_core_maturity_orig", [EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/original_raw.npz" for f in range(5)], key="prediction")
    add_model("tr_core_maturity_fresh", [EXP_ROOT / f"transfer-20261025/core_maturity/fold{f}/fresh_raw.npz" for f in range(5)], key="prediction")
    add_model("tr_early_avg_raw_mean", [EXP_ROOT / f"transfer-20261025/early_average/fold{f}/raw_mean.npz" for f in range(5)], key="prediction")
    add_model("tr_raw_seed_repeat", [EXP_ROOT / f"transfer-20261025/robustness/fold{f}/raw_seed_repeat.npz" for f in range(5)], key="prediction")
    add_model("tr_spline_query", [EXP_ROOT / f"transfer-20261025/joint_correction/fold{f}/spline_query.npz" for f in range(5)], key="prediction")
    for sub in ["balanced__original", "balanced__fresh_kumo_seed", "proportional__original", "proportional__fresh_kumo_seed"]:
        add_model(f"tr_{sub}", [EXP_ROOT / f"transfer-20261025/balanced_context/fold{f}/{sub}/raw_predictions.npz" for f in range(5)], key="prior_adjusted_probability")

    for seed in [2026, 2027]:
        for hname in ["account_dynamic", "account_spline", "bureau_dynamic", "bureau_logit", "champion0", "champion1", "champion2", "champion3", "champion4", "pricing_aux", "shopping_xgb"]:
            add_model(f"h19_s{seed}_{hname}", [EXP_ROOT / f"horizontal-20261019/campaign/fits/s{seed}_f{f}_context_{hname}/prediction.npz" for f in range(5)], key="prediction")

    for size in ["large8", "medium8"]:
        sdir = Path(f"/mnt/home/faris/logika-horizontal-20261026/runs/{size}")
        if sdir.exists():
            for pack in ["baseline90", "p0018"]:
                add_model(f"horiz_{size}_{pack}", [sdir / pack / f"fold{f}/prediction.npz" for f in range(5)])

    nested_arrs = []
    lin56 = np.load(OUT_ROOT / "linear56k_runs.npz", allow_pickle=True)
    sp200_arrs = []
    d56_glob_arrs = []
    d56_pop_arrs = []
    cpu_keys = ["corr_exp_unweighted", "corr_exp_boundary", "res_lgbm_56k", "res_cat_56k", "full_lgbm_56k", "full_cat_56k"]
    cpu_arrs = {ck: [] for ck in cpu_keys}

    for f in range(5):
        r = roles_list[f]
        q = np.r_[r["correction"], r["validation"]]
        meta = np.load(EXP_ROOT / f"nested-correction-20261021/heads/fold{f}/meta_inputs.npz", allow_pickle=True)
        raw5 = np.load(EXP_ROOT / f"nested-correction-20261021/runs/job{f*6+5}/raw.npz", allow_pickle=True)
        pk = np.full(70000, np.nan, dtype=np.float64)
        pk[meta["rows"]] = meta["prediction"]
        pk[raw5["rows"]] = raw5["prediction"]
        nested_arrs.append(pk[q])

        sp200_arrs.append(np.r_[lin56["oof_spline200"][f][r["correction"]], lin56["val_spline200"][r["validation"]]])
        d56_glob_arrs.append(np.r_[expit(logit(clip_prob(pk[r["correction"]])) + 0.5 * lin56["oof_delta56_glob"][f][r["correction"]]),
                                    expit(logit(clip_prob(pk[r["validation"]])) + 0.5 * lin56["val_delta56_glob"][r["validation"]])])
        d56_pop_arrs.append(np.r_[expit(logit(clip_prob(pk[r["correction"]])) + 0.5 * lin56["oof_delta56_pop"][f][r["correction"]]),
                                   expit(logit(clip_prob(pk[r["validation"]])) + 0.5 * lin56["val_delta56_pop"][r["validation"]])])

        cpu_f = np.load(OUT_ROOT / f"cpu_runs/fold{f}_crossfit56k.npz", allow_pickle=True)
        for ck in cpu_keys:
            cpu_arrs[ck].append(np.r_[cpu_f[f"oof_{ck}"][50000:], cpu_f[f"val_{ck}"]])

    models["nested_kumo56"] = nested_arrs
    models["spline200_56k"] = sp200_arrs
    models["kumo56_delta_glob"] = d56_glob_arrs
    models["kumo56_delta_pop"] = d56_pop_arrs
    for ck in cpu_keys:
        models[f"cpu_{ck}"] = cpu_arrs[ck]

    return models, lin56

models, lin56 = load_20k_cohort()
print(f"Total 20k-aligned models (including 56k OOF-aligned & CPU cross-fits): {len(models)}", flush=True)

def avg_logit(names, f):
    valid = [n for n in names if n in models]
    if not valid:
        raise ValueError(f"No valid models in {names}")
    return np.mean([logit(clip_prob(models[n][f])) for n in valid], axis=0)

aug_l8_names = [k for k in models if k.startswith("aug90_large8_")]
aug_m8_names = [k for k in models if k.startswith("aug90_medium8_")]
fs_l_names = [k for k in models if k.startswith("fs_kumo_large_")]
fs_m_names = [k for k in models if k.startswith("fs_kumo_medium_")]
popA_kumo = ["kumo_covplace_coverage96", "fe_coverage_info", "aug90_medium8_repayment_credibility", "aug90_large8_volatility", "tr_proportional__original", "kumo_covplace_control90"]
popB_kumo = ["tr_proportional__original", "tr_balanced__fresh_kumo_seed", "tr_proportional__fresh_kumo_seed", "tr_balanced__original", "aug90_large8_repayment_credibility"]
popC1_kumo = ["fs_kumo_medium_latent99", "fs_kumo_medium_full100", "fs_kumo_medium_latent99_s7731", "fs_kumo_large_full100_s7731", "fs_kumo_medium_shadow91", "aug90_large8_liquidity_payment"]
popC2_kumo = ["joint_kumo_pop_specialist", "horiz_large8_baseline90", "horiz_large8_p0018", "fe_mature_mask_preserved", "kumo_supervised90", "nested_kumo56", "fs_kumo_large_champion90_s7731"]
tr_kumo = ["tr_early_avg_raw_mean", "tr_raw_seed_repeat", "tr_core_maturity_orig", "tr_core_maturity_fresh", "hmf_classification"]
icl_other = ["arena_TabSwift_config0", "arena_TabICLv2_config0", "tabicl_compact"]
neural_names = [k for k in models if k.startswith("arch_tabm_") or k.startswith("arch_resnet_") or k.startswith("arch_dcn_")]
gbdt_ctx_names = ["fs_catboost_master", "fs_lgbm_master", "arena_CatBoost_config0", "arena_CatBoost_config1", "arena_CatBoost_config2", "arena_LightGBM_config0", "arena_XGBoost_config0"]
h19_champions = [f"h19_s{s}_champion{c}" for s in [2026, 2027] for c in range(5)]

families_spec = {
    "pop_spec": ["joint_kumo_pop_specialist"],
    "surg96": ["joint_kumo_surgical96"] if "joint_kumo_surgical96" in models else ["joint_kumo_pop_specialist"],
    "aug_l8": aug_l8_names,
    "aug_m8": aug_m8_names,
    "fs_large": fs_l_names,
    "fs_med": fs_m_names,
    "popA_kumo": popA_kumo,
    "popB_kumo": popB_kumo,
    "popC1_kumo": popC1_kumo,
    "popC2_kumo": popC2_kumo,
    "tr_kumo": tr_kumo,
    "h19_champ": h19_champions,
    "nested_kumo56": ["nested_kumo56"],
    "kumo56_dglob": ["kumo56_delta_glob"],
    "kumo56_dpop": ["kumo56_delta_pop"],
    "cpu_corr_bnd": ["cpu_corr_exp_boundary"],
    "cpu_res_lgbm": ["cpu_res_lgbm_56k"],
    "cpu_res_cat": ["cpu_res_cat_56k"],
    "cpu_full_lgbm": ["cpu_full_lgbm_56k"],
    "cpu_full_cat": ["cpu_full_cat_56k"],
    "icl_other": icl_other,
    "neural": neural_names,
    "gbdt_ctx": gbdt_ctx_names,
    "spline200": ["spline200_56k"],
}

fam_cor_val = {fname: [avg_logit(fmodels, f) for f in range(5)] for fname, fmodels in families_spec.items()}

print("\n=== FAMILY PERFORMANCE ON 30K CORRECTION & 70K OUTER VALIDATION ===", flush=True)
y_cor_all = np.concatenate([roles_list[f]["y_correction"] for f in range(5)])
for fname, farrs in fam_cor_val.items():
    cor = np.concatenate([expit(a[:6000]) for a in farrs])
    val = np.zeros(70000, dtype=np.float64)
    for f in range(5):
        val[roles_list[f]["validation"]] = expit(farrs[f][6000:])
    print(f"  {fname:16s}: CorKS={ks_stat(y_cor_all, cor):.6f} CorAUC={roc_auc_score(y_cor_all, cor):.6f} | ValKS={ks_stat(y_global, val):.6f} ValAUC={roc_auc_score(y_global, val):.6f}", flush=True)

# 3. Evaluate exact feature_head + tabular/spline blend on our new ensembles!
raw_npz = np.load(OUT_ROOT / "raw_domain_features.npz", allow_pickle=True)
raw_names = [str(x) for x in raw_npz["names"]]
raw_data = raw_npz["data"]
pop_masks = {p: raw_data[:, raw_names.index(p)] > 0.5 for p in ["pop_A", "pop_B", "pop_C1", "pop_C2"]}

print("\n=== TESTING EXACT FEATURE_HEAD + TABULAR/SPLINE BLENDS ===", flush=True)
for w_spec in [0.15, 0.25, 0.35]:
    for w_res56 in [0.0, 0.15, 0.25, 0.35]:
        val_raw_ens = np.zeros(70000, dtype=np.float64)
        val_head18_ens = np.zeros(70000, dtype=np.float64)
        val_orig_sp_blend = np.zeros(70000, dtype=np.float64)
        val_up_tab_blend = np.zeros(70000, dtype=np.float64)
        cor_raw_ens = []

        for f in range(5):
            r = roles_list[f]
            q_idx = np.r_[r["correction"], r["validation"]]
            mA = pop_masks["pop_A"][q_idx]
            mB = pop_masks["pop_B"][q_idx]
            mC1 = pop_masks["pop_C1"][q_idx]
            mC2 = pop_masks["pop_C2"][q_idx]

            z_pop_routed = np.zeros(20000, dtype=np.float64)
            z_pop_routed[mA] = 0.45 * fam_cor_val["popA_kumo"][f][mA] + 0.30 * fam_cor_val["aug_l8"][f][mA] + 0.25 * fam_cor_val["surg96"][f][mA]
            z_pop_routed[mB] = 0.50 * fam_cor_val["popB_kumo"][f][mB] + 0.30 * fam_cor_val["aug_l8"][f][mB] + 0.20 * fam_cor_val["pop_spec"][f][mB]
            z_pop_routed[mC1] = 0.45 * fam_cor_val["popC1_kumo"][f][mC1] + 0.30 * fam_cor_val["aug_l8"][f][mC1] + 0.25 * fam_cor_val["pop_spec"][f][mC1]
            z_pop_routed[mC2] = 0.45 * fam_cor_val["popC2_kumo"][f][mC2] + 0.30 * fam_cor_val["aug_l8"][f][mC2] + 0.25 * fam_cor_val["fs_large"][f][mC2]

            # Multi-view 50k context logit
            z_ctx50 = (
                w_spec * z_pop_routed
                + (0.35 - w_spec) * fam_cor_val["pop_spec"][f]
                + 0.20 * fam_cor_val["aug_l8"][f]
                + 0.14 * fam_cor_val["fs_large"][f]
                + 0.08 * fam_cor_val["aug_m8"][f]
                + 0.06 * fam_cor_val["fs_med"][f]
                + 0.06 * fam_cor_val["tr_kumo"][f]
                + 0.05 * fam_cor_val["h19_champ"][f]
                + 0.03 * fam_cor_val["icl_other"][f]
                + 0.03 * fam_cor_val["neural"][f]
            )

            # 56k cross-fitted residual-boosted Kumo logit
            z_res56 = (
                0.35 * fam_cor_val["cpu_corr_bnd"][f]
                + 0.25 * fam_cor_val["kumo56_dpop"][f]
                + 0.20 * fam_cor_val["cpu_res_lgbm"][f]
                + 0.20 * fam_cor_val["cpu_res_cat"][f]
            )

            z_comb = (1.0 - w_res56) * z_ctx50 + w_res56 * z_res56
            p_comb = expit(z_comb)
            cor_raw_ens.append(p_comb[:6000])
            val_raw_ens[r["validation"]] = p_comb[6000:]

            # Apply exact feature_head on 6,000 correction rows!
            head = feature_head.fit(tables_20k[f].iloc[:6000].assign(target=r["y_correction"]), p_comb[:6000], HEAD18_COLS, 0.003, 0.5)
            p_head_val = b_mod.corrected_probability(head, tables_20k[f].iloc[6000:], p_comb[6000:])[0]
            val_head18_ens[r["validation"]] = p_head_val
            val_orig_sp_blend[r["validation"]] = 0.75 * p_head_val + 0.25 * baseline_spline_val[f]

            # Upgraded tabular/spline branch
            p_tab_val = (
                0.35 * baseline_spline_val[f]
                + 0.25 * expit(fam_cor_val["spline200"][f][6000:])
                + 0.20 * expit(fam_cor_val["gbdt_ctx"][f][6000:])
                + 0.10 * expit(fam_cor_val["cpu_full_lgbm"][f][6000:])
                + 0.10 * expit(fam_cor_val["cpu_full_cat"][f][6000:])
            )
            val_up_tab_blend[r["validation"]] = 0.75 * p_head_val + 0.25 * p_tab_val

        cor_all = np.concatenate(cor_raw_ens)
        print(f"[w_spec={w_spec:.2f} w_res56={w_res56:.2f}] CorRawKS={ks_stat(y_cor_all, cor_all):.6f} | ValRawKS={ks_stat(y_global, val_raw_ens):.6f} | ValHead18KS={ks_stat(y_global, val_head18_ens):.6f} | ValOrigSpKS={ks_stat(y_global, val_orig_sp_blend):.6f} | ValUpTabKS={ks_stat(y_global, val_up_tab_blend):.6f} (AUC={roc_auc_score(y_global, val_up_tab_blend):.6f})", flush=True)

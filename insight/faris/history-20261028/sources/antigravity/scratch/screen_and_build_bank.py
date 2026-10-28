import numpy as np
import pandas as pd
import json
import time

t0 = time.time()

# 1. Load global y and roles
y_global = np.zeros(70000, dtype=np.float64)
roles_list = []
p_inner_list = []
for f in range(5):
    r = np.load(f"/mnt/home/faris/logika-frozen-shadow-20261026/frozen/fold{f}/roles.npz", allow_pickle=True)
    roles_list.append(r)
    y_global[r["context"]] = r["y_context"]
    y_global[r["correction"]] = r["y_correction"]
    meta = np.load(f"/mnt/home/faris/logika-faris-exploration/nested-correction-20261021/heads/fold{f}/meta_inputs.npz", allow_pickle=True)
    p_in = np.full(70000, np.nan, dtype=np.float64)
    p_in[meta["rows"]] = meta["prediction"]
    p_inner_list.append(p_in)

# 2. Load raw_domain_features.npz and extra.parquet
raw_npz = np.load("/mnt/home/faris/logika-joint-20261028/raw_domain_features.npz", allow_pickle=True)
raw_names = [str(x) for x in raw_npz["names"]]
raw_data = raw_npz["data"].astype(np.float32)
assert np.array_equal(y_global.astype(int), raw_npz["y"].astype(int)), "y_global mismatch with raw_npz!"

extra_df = pd.read_parquet("/mnt/home/faris/logika-horizontal-20261026/extra.parquet")
extra_names = [f"hz_{c}" for c in extra_df.columns]
extra_data = extra_df.values.astype(np.float32)

# Check which columns in catboost-convergence fold0 are invariant across folds vs fold-dependent
cb0_tr = pd.read_parquet("/mnt/home/faris/logika-faris-exploration/catboost-convergence-20261021/data/fold0/train.parquet")
cb0_va = pd.read_parquet("/mnt/home/faris/logika-faris-exploration/catboost-convergence-20261021/data/fold0/validation.parquet")
cat_cols = ["loan_purpose", "region", "employment_type", "housing_status", "target", "application_id"]
cb_num_cols = [c for c in cb0_tr.columns if c not in cat_cols and c in cb0_va.columns and pd.api.types.is_numeric_dtype(cb0_tr[c])]
cb_str_cols = [c for c in cb0_tr.columns if c not in ["target", "application_id"] and c in cb0_va.columns and not pd.api.types.is_numeric_dtype(cb0_tr[c])]
print("CB string cols:", cb_str_cols)

# Reconstruct global 70k matrix for fold 0 from cb0
def cb_to_global(f, cols):
    r = roles_list[f]
    tr_idx = np.concatenate([r["context"], r["correction"]])
    va_idx = r["validation"]
    df_tr = pd.read_parquet(f"/mnt/home/faris/logika-faris-exploration/catboost-convergence-20261021/data/fold{f}/train.parquet", columns=cols)
    df_va = pd.read_parquet(f"/mnt/home/faris/logika-faris-exploration/catboost-convergence-20261021/data/fold{f}/validation.parquet", columns=cols)
    out = np.empty((70000, len(cols)), dtype=np.float32)
    out[tr_idx] = df_tr.values.astype(np.float32)
    out[va_idx] = df_va.values.astype(np.float32)
    return out

print("Loading cb fold 0 and fold 1 to identify invariant vs fold-specific columns...")
cb_g0 = cb_to_global(0, cb_num_cols)
cb_g1 = cb_to_global(1, cb_num_cols)

diff01 = np.nanmax(np.abs(cb_g0 - cb_g1), axis=0)
invariant_mask = (diff01 < 1e-6) | np.all(np.isnan(cb_g0), axis=0)
inv_cols = [cb_num_cols[i] for i in range(len(cb_num_cols)) if invariant_mask[i]]
fold_spec_cb_cols = [cb_num_cols[i] for i in range(len(cb_num_cols)) if not invariant_mask[i]]
print(f"CB numeric cols: {len(cb_num_cols)} total -> {len(inv_cols)} invariant, {len(fold_spec_cb_cols)} fold-specific")
print("Sample fold-specific CB cols:", fold_spec_cb_cols[:20])

# Now screen all invariant CB cols + raw_domain_features + extra_df against the EXACT 56,000 inner-OOF Kumo residuals across all 5 folds!
all_inv_names = inv_cols + raw_names + extra_names
all_inv_mat = np.hstack([cb_g0[:, invariant_mask], raw_data, extra_data])
print(f"Total invariant candidate features: {len(all_inv_names)} ({all_inv_mat.shape})")

# Compute residual Z-stats across the 5 folds using ONLY the 56,000 inner-OOF training rows of each fold!
pop_A = raw_data[:, raw_names.index("pop_A")] > 0.5
pop_B = raw_data[:, raw_names.index("pop_B")] > 0.5
pop_C1 = raw_data[:, raw_names.index("pop_C1")] > 0.5
pop_C2 = raw_data[:, raw_names.index("pop_C2")] > 0.5

# Standardize features globally (with median fill and log-compression for skewed features) for fast correlation screening
X = all_inv_mat.astype(np.float64).copy()
nan_mask = np.isnan(X)
# Also create missingness indicators for columns with >= 1% and <= 99% missing
miss_rate = nan_mask.mean(axis=0)
q05 = np.nanpercentile(X, 5, axis=0)
q50 = np.nanmedian(X, axis=0)
q95 = np.nanpercentile(X, 95, axis=0)
# Log-compress skewed positive columns
skew_cols = (q05 >= 0) & ((q95 - q50) > 4.0 * np.maximum(q50 - q05, 1e-6)) & (q95 > 10.0)
X[:, skew_cols] = np.log1p(np.maximum(X[:, skew_cols], 0.0))
q50_new = np.nanmedian(X, axis=0)
q50_new = np.where(np.isnan(q50_new), 0.0, q50_new)
inds = np.where(nan_mask)
X[inds] = np.take(q50_new, inds[1])
mu = X.mean(axis=0)
sd = X.std(axis=0)
sd = np.where(sd < 1e-8, 1.0, sd)
X = np.clip((X - mu) / sd, -6.0, 6.0)

def compute_fold_z(sub_mask_fn):
    corrs = np.zeros((5, X.shape[1]), dtype=np.float64)
    for f in range(5):
        r = roles_list[f]
        tr_idx = np.concatenate([r["context"], r["correction"]])
        m = np.zeros(70000, dtype=bool)
        m[tr_idx] = True
        m = m & sub_mask_fn(f, p_inner_list[f])
        idx = np.where(m)[0]
        res = y_global[idx] - p_inner_list[f][idx]
        res = res - res.mean()
        res_sd = res.std()
        Xf = X[idx]
        Xf_c = Xf - Xf.mean(axis=0)
        Xf_sd = np.maximum(Xf_c.std(axis=0), 1e-8)
        # Studentized correlation: r * sqrt(n)
        corrs[f] = ((Xf_c * res[:, None]).mean(axis=0) / (Xf_sd * res_sd)) * np.sqrt(len(idx))
    mean_z = corrs.mean(axis=0)
    same_sign = np.maximum((corrs > 0).sum(axis=0), (corrs < 0).sum(axis=0))
    return mean_z, same_sign, corrs

z_all, sign_all, _ = compute_fold_z(lambda f, p: np.ones(70000, dtype=bool))
z_band, sign_band, _ = compute_fold_z(lambda f, p: (p >= 0.05) & (p <= 0.35))
z_popA, sign_popA, _ = compute_fold_z(lambda f, p: pop_A)
z_popB, sign_popB, _ = compute_fold_z(lambda f, p: pop_B)
z_popC1, sign_popC1, _ = compute_fold_z(lambda f, p: pop_C1)
z_popC2, sign_popC2, _ = compute_fold_z(lambda f, p: pop_C2)

# Also check fold-specific columns from frozen-shadow and cb_num_cols!
frz0 = pd.read_parquet("/mnt/home/faris/logika-frozen-shadow-20261026/frozen/fold0/features.parquet")
champ90_cols = list(frz0.columns[:90])
head18_cols = list(frz0.columns[90:106])
shadow_latent_cols = list(frz0.columns[106:])
existing_set = set(champ90_cols + head18_cols)

# Screen fold-specific shadow_latent_cols + fold_spec_cb_cols across all 5 folds
fspec_corrs = np.zeros((5, len(shadow_latent_cols) + len(fold_spec_cb_cols)), dtype=np.float64)
fspec_names = shadow_latent_cols + fold_spec_cb_cols
for f in range(5):
    r = roles_list[f]
    tr_idx = np.concatenate([r["context"], r["correction"]])
    frzf = pd.read_parquet(f"/mnt/home/faris/logika-frozen-shadow-20261026/frozen/fold{f}/features.parquet", columns=shadow_latent_cols)
    cbf_tr = pd.read_parquet(f"/mnt/home/faris/logika-faris-exploration/catboost-convergence-20261021/data/fold{f}/train.parquet", columns=fold_spec_cb_cols)
    mat_f = np.hstack([frzf.values[tr_idx].astype(np.float64), cbf_tr.values.astype(np.float64)])
    med_f = np.nanmedian(mat_f, axis=0)
    med_f = np.where(np.isnan(med_f), 0.0, med_f)
    inds_f = np.where(np.isnan(mat_f))
    mat_f[inds_f] = np.take(med_f, inds_f[1])
    mat_c = mat_f - mat_f.mean(axis=0)
    mat_sd = np.maximum(mat_c.std(axis=0), 1e-8)
    res = y_global[tr_idx] - p_inner_list[f][tr_idx]
    res = res - res.mean()
    res_sd = res.std()
    fspec_corrs[f] = ((mat_c * res[:, None]).mean(axis=0) / (mat_sd * res_sd)) * np.sqrt(len(tr_idx))

print("\n=== FOLD-SPECIFIC FEATURES (SHADOW / LATENT / CB FOLD-SPECIFIC) BY |Z_ALL| ===")
for j, fn in enumerate(fspec_names):
    print(f"  {fn:36s}: z_all={fspec_corrs[:, j].mean():+.3f} (folds: {np.round(fspec_corrs[:, j], 2).tolist()})")

rows_summary = []
for i, name in enumerate(all_inv_names):
    rows_summary.append({
        "name": name,
        "in_champ_or_head": name in existing_set,
        "z_all": float(z_all[i]),
        "sign_all": int(sign_all[i]),
        "z_band": float(z_band[i]),
        "z_popA": float(z_popA[i]),
        "z_popB": float(z_popB[i]),
        "z_popC1": float(z_popC1[i]),
        "z_popC2": float(z_popC2[i]),
        "max_abs_z": float(max(abs(z_all[i]), abs(z_band[i]), abs(z_popA[i]), abs(z_popB[i]), abs(z_popC1[i]), abs(z_popC2[i]))),
    })

df_sum = pd.DataFrame(rows_summary)
df_new = df_sum[~df_sum["in_champ_or_head"]].copy()

print("\n=== TOP 35 NEW FEATURES BY |Z_ALL| (56,000 INNER-OOF KUMO RESIDUALS ACROSS 5 FOLDS) ===")
print(df_new.reindex(df_new["z_all"].abs().sort_values(ascending=False).index).head(35)[["name", "z_all", "sign_all", "z_band", "z_popA", "z_popB", "z_popC1", "z_popC2"]].to_string(index=False))

print("\n=== TOP 20 NEW FEATURES FOR POP A (NO ACCOUNT/PREV/REPAY, 30% OF DATA) ===")
print(df_new.reindex(df_new["z_popA"].abs().sort_values(ascending=False).index).head(20)[["name", "z_popA", "z_all", "z_band"]].to_string(index=False))

print("\n=== TOP 20 NEW FEATURES FOR POP B (REJECTED/CANCELLED PREV, NO REPAY, 3.84% OF DATA) ===")
print(df_new.reindex(df_new["z_popB"].abs().sort_values(ascending=False).index).head(20)[["name", "z_popB", "z_all", "z_band"]].to_string(index=False))

print("\n=== TOP 20 NEW FEATURES FOR POP C1 (SHORT REPAY 1..11 INST, 16.04% OF DATA) ===")
print(df_new.reindex(df_new["z_popC1"].abs().sort_values(ascending=False).index).head(20)[["name", "z_popC1", "z_all", "z_band"]].to_string(index=False))

print("\n=== TOP 20 NEW FEATURES FOR POP C2 (MATURE REPAY >=12 INST, 50.12% OF DATA) ===")
print(df_new.reindex(df_new["z_popC2"].abs().sort_values(ascending=False).index).head(20)[["name", "z_popC2", "z_all", "z_band"]].to_string(index=False))

df_sum.to_csv("/mnt/home/faris/logika-joint-20261028/inner_oof_residual_screen.csv", index=False)
print(f"\nCompleted exact 56k inner-OOF residual screen in {time.time()-t0:.1f}s")

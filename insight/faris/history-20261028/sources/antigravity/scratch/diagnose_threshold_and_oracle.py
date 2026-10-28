from pathlib import Path
import numpy as np

OUT_ROOT = Path("/mnt/home/faris/logika-joint-20261028")
EXP_ROOT = Path("/mnt/home/faris/logika-faris-exploration")
FRZ_ROOT = Path("/mnt/home/faris/logika-frozen-shadow-20261026")

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

print(f"Global base rate: {y_global.mean():.5f} (N={len(y_global)}, P={y_global.sum()})")
for pn, m in pop_masks.items():
    print(f"  {pn:8s}: n={m.sum():5d} ({m.mean()*100:5.1f}%), pos={y_global[m].sum():4d}, base_rate={y_global[m].mean():.5f}, N_g/N={np.sum(y_global[m]==0)/np.sum(y_global==0):.4f}, P_g/P={np.sum(y_global[m]==1)/np.sum(y_global==1):.4f}")

print("\nPer-fold base rates (context / correction / validation):")
for f in range(5):
    r = np.load(FRZ_ROOT / f"frozen/fold{f}/roles.npz", allow_pickle=True)
    y_ctx = y_global[r["context"]]
    y_cor = y_global[r["correction"]]
    y_val = y_global[r["validation"]]
    print(f"  Fold {f}: ctx={y_ctx.mean():.5f}, cor={y_cor.mean():.5f}, val={y_val.mean():.5f} | base_raw_cor_mean={np.load(FRZ_ROOT / f'frozen/fold{f}/baseline.npz')['raw'][:6000].mean():.5f}, base_final_val_mean={baseline_oof[r['validation']].mean():.5f}")

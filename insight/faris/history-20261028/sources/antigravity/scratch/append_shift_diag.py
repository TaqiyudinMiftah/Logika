from pathlib import Path
import numpy as np
from scipy.special import expit, logit

# Let's add a print of mean_c, pi_g_56k, and t_o in test_cor_threshold_shift.py
with open("/home/parasite/.gemini/antigravity-cli/brain/73181457-58b8-405a-bb23-a8afbd02f051/scratch/test_cor_threshold_shift.py") as f:
    code = f.read()

extra = """
for pn in ["pop_A", "pop_B", "pop_C1", "pop_C2"]:
    m = pop_masks[pn]
    print(f"Subgroup {pn:6s}: mean_pred_val={raw_assembled_val[m].mean():.5f}, true_base_rate={y_global[m].mean():.5f}, oracle_t={oracle_ts[pn]:.5f}")

for lam in np.linspace(0.0, 1.5, 31):
    sh = np.zeros(70000)
    for f in range(5):
        r = roles_list[f]
        c_idx = r["correction"]
        v_idx = r["validation"]
        pi_f = float(r["y_correction"].mean())
        for pn, mod_list in recipes.items():
            mc = pop_masks[pn][c_idx]
            mv = pop_masks[pn][v_idx]
            p_c = np.mean([cands_cor[m][f][mc] for m in mod_list], axis=0)
            p_v = raw_assembled_val[v_idx[mv]]
            z_v = logit(np.clip(p_v, 1e-6, 1.0 - 1e-6))
            mean_c = float(np.mean(p_c))
            sh[v_idx[mv]] = expit(z_v - lam * (logit(mean_c) - logit(pi_f)))
    print(f"  lam={lam:.2f} -> KS={ks_stat(y_global, sh):.6f}")
"""

with open("/home/parasite/.gemini/antigravity-cli/brain/73181457-58b8-405a-bb23-a8afbd02f051/scratch/test_cor_threshold_shift.py", "w") as f:
    f.write(code + "\n" + extra)

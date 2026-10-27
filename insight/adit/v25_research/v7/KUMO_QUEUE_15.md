# Antrean15 eksperimen Kumo100% frozen

Selesai10/15. BaselineV25; target pooledKS>=0.531. Auto-start saat resource cukup. Cache-only V172/V177 dan adapterV179â€“V183 tidak memerlukan GPU.

| ID | Metode | Status | KS | AUC |
|---|---|---|---:|---:|
| V169 | Kumo Large learns V25 exposure and conditional-liquidity counts directly | COMPLETE | 0.5288125269068431 | 0.8472206798279507 |
| V170 | Frozen Kumo with spending velocity and liquidity pressure | COMPLETE | 0.5289414385812214 | 0.8472102261201847 |
| V171 | Frozen Kumo with compact spectral band representation | COMPLETE | 0.5293593991007075 | 0.8471952967231657 |
| V172 | Coverage-aware spectral model with V25 fallback for unobserved repayment allocation | COMPLETE | 0.5303631341433743 | 0.8471641199134107 |
| V173 | Stratified context bagging: two40k frozen Kumo fits from identical50k eligible pool | COMPLETE | 0.5286022721015591 | 0.8470521268312751 |
| V174 | Empirical-Bayes allocation propensity and uncertainty inside frozen Kumo | COMPLETE | 0.5292927367174179 | 0.8469899262267827 |
| V175 | Selective latent uncertainty and income-contradiction inputs | COMPLETE | 0.5284933723571009 | 0.8472016433687654 |
| V176 | Compact item-value basket composition, without unverified order denominators | COMPLETE | 0.529325332393696 | 0.8469974152685905 |
| V177 | Conservative fixed90:10 V25 and V169 score blend | COMPLETE | 0.5303371946203446 | 0.847059735850767 |
| V178 | Conservative global-local frozen context experts by history coverage | COMPLETE | 0.5305973914406749 | 0.8471293837656989 |
| V179 | Group-ridge spectral correction: preserve V25 penalty, shrink new family10x | QUEUED | BELUM TERSEDIA | BELUM TERSEDIA |
| V180 | Reliability-weighted spectral residual with frozen V25 correction | QUEUED | BELUM TERSEDIA | BELUM TERSEDIA |
| V181 | Fixed equal average of V167 spectral and V146 tree-complemented V25 | QUEUED | BELUM TERSEDIA | BELUM TERSEDIA |
| V182 | 64-landmark Nystrom RBF residual on existing V25 domain features | QUEUED | BELUM TERSEDIA | BELUM TERSEDIA |
| V183 | Strongly regularized shallow LightGBM residual on fixed V25 logits | QUEUED | BELUM TERSEDIA | BELUM TERSEDIA |
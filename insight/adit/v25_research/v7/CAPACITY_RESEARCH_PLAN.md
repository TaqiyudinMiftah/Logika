# Capacity controls after statistical/stability experiments

All3 candidates are executable recipes, but readiness remainsfalse until pinned Small weights and worker checks pass. Kumo is100%frozen; fit means in-context cache construction, not gradient fine-tuning. Same90 inputs, identical outer folds/context/correction allocations and original spline. No public score is transferred from older Small submissions.

## V207: Frozen Small90 capacity control with exact V25 folds

All previous new inputs retained the same Large backbone and yielded small or unstable gains. Test a substantive reduction of pretrained capacity with identical data information, folds and adaptation settings. Earlier public Small results used different pipelines and are not transferred. Compare fresh LargeV168 as runtime control.

Formula: `KumoSmall100%frozen, same90 inputs/50kcontext/seed2026+fold/8estimators; original18-input6k correction L2.003 shrink.5; same77-input spline and75:25fusion; mean5foldtest`

## V208: Frozen Small plus stable9 correction information

Separate capacity from representation by reusing identical Small scores and only adding9 stable power/support inputs. The Large analogV196 improves AUC without KS; Small may need complementary explicit temporal information, but benefit is unproved.

Formula: `ReuseV207 frozenSmall context predictions; fit27-input correction adding stable9 spectral inputs, L2.003/shrink.5; retain original75:25spline77fusion`

## V209: Conservative Small/Large capacity-diversity blend

A smaller frozen expert can have different ranking errors despite lower standalone performance. Test limited complementary information, without fitting blend weights on outer labels; no public-score gain assumed.

Formula: `Perfold final=.90*pV193+.10*pV207, fixed weights before seeingSmall results; test probabilities averaged after per-fold blending`


## Refill based on early stability results

### V210: V193 anchor with10% high-AUC reduced-knot representation

V198 AUC.84726014 exceedsV193 by.00015260 but its KS.52937399 is lower. Test a prespecified small contribution to broader ranking without replacing the stronger KS model. This is a new representation source, not an outer-fitted blend search.

Formula: `Perfold p=.90*pV193+.10*pV198; fixed weights, then mean5foldtest`

### V211: Frozen Small original18 correction with reduced knots

Separate correction capacity from backbone capacity. Small may require a different residual complexity, but this remains a hypothesis. Context predictions and features are identical toV207.

Formula: `Same cachedSmall90/50kcontext asV207,18-input head with quadratic2knots ratherthan3, L2.003/shrink.5; same75:25spline77fusion`

### V212: Coverage-aware conservative Small/Large blend

V185 spectral contribution lost separation in no-allocation borrowers, while only one Large capacity was used. Test a bounded Small contribution larger for thin internal histories. Do not assume Small is safer: compare public-incumbent and development metrics, with explicit missing-data caveat.

Formula: `wSmall=.15*10/(max(finite_allocation_count,0)+10); p=(1-wSmall)*pV193+wSmall*pV207; missing count getsw.15; mean5foldtest`


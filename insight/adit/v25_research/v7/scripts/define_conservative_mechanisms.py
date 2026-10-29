"""Write the prespecified five-experiment plan; activation is separate."""
from pathlib import Path
import csv,json
V=Path(__file__).resolve().parents[1]
cols=json.loads((V/'anchored_spline_plan.json').read_text())[1]['input_features']
common=dict(worker='conservative_mechanisms.py',source_script='v7/scripts/conservative_mechanisms.py',research_ready=False,input_features=cols,Kumo_frozen_fraction=1.,comparison_baseline='V25',execution_host='local_windows',priority='P1',new_weight=.05,no_outer_validation_tuning=True,preflight_receipt='v7/conservative_mechanisms/reports/PREFLIGHT.json',head_gap_caveat='Correction-head gap is descriptive; for monotone standalone expert the original gap is unchanged and separate expert training/heldout metrics are provided.')
items=[
('V257','Posterior uncertainty integration of fixed correction','posterior_quadrature',
 'Independent normal working distributions for price_log and bureau latent means with their existing posteriorSD. Evaluate frozen V25 correction at mu+sqrt(2)*SD*t over3x3 Gauss-Hermite nodes, normalized product weights; no refit. p_corrected=.95*p_V25_corrected+.05*E[p_corrected(draw)]; final=.75*p_corrected+.25*originalSpline77. Missing mean stays missing, missingSD gives point estimate; nonnegativeSD asserted.',
 'V25 uses posterior means plus SD as separate inputs but evaluates nonlinear effects at a point. Test marginalization of uncertain measurements, especially thin histories. Different from V202/V203 Laplace coefficient uncertainty and peer tensor mean-times-SD interactions. Independent Gaussian posteriors are a working approximation, not verified joint distributions; old global-reference provenance is inherited.'),
('V258','Leave-one-source-out correction consensus','source_jackknife',
 'Fit four original ridge.003 strength.5 heads on identical6k, each omitting exactly one of pricing(2), bureau(2), capacity(5), repayment(2) latent-feature blocks. Retain six coverage columns and employment. q=equal mean of four predictions; p_corrected=.95*p_V25_corrected+.05*q; final=.75*p_corrected+.25*originalSpline77. Context/backbone unchanged.',
 'Added histories/complexity repeatedly lowered AUC; test sensitivity to reliance on individual noisy views without changing Kumo context. V187/V219 resampled rows, whereas this ensemble changes correction information blocks. Not an ablation of the whole source because Kumo already contains those sources. Every member uses same6k and remains correlated.'),
('V259','Function-variance regularized correction','function_space_ridge',
 'Original V25 design X. For each input block j including its missing indicator, Cj=Xj-columnmean_train(Xj). Fit meanBCE(z+X*beta)+.003/2||beta_nonintercept||^2+.01/(2n)*sum_j||Cj*beta_j||^2 on6k. Offset-slope gets original ridge only; intercept unpenalized. Predict q=sigmoid(z+.5X*beta); p_corrected=.95*p_V25_corrected+.05*q; final75:25originalSpline.',
 'Noise audit identified duplicated coverage indicators and parameterization-dependent ridge. This adds a penalty on each learned feature effect variance, rather than coefficient differences(V200), low-rank compression(V240), or row leverage(V241). Existing ridge remains, so this is not claimed fully basis-invariant. Test reducing unstable amplitude without deleting risk information.'),
('V260','Small monotone credit-capacity expert','monotone_credit_expert',
 'On56k outer-training rows fit a logistic additive model on four signedlog inputs: negativebureau, late30 propensity, installment/observed income, existing totalburden. Training quantiles[0,.25,.5,.75,1] define clipped unit ramps; nonnegative ramp coefficients, free intercept/missing flags, meanBCE+.01/2||nonintercept||^2. No categorical input. Final=.95*p_V25+.05*p_expert; fixed folds and no outer early stopping.',
 'V254-256 test unrestricted small spline experts; this distinct domain-constrained expert tests whether limiting reversal of credit/capacity effects improves robustness. Monotonic directions are financial hypotheses, not established causal laws, and may fail due to selection/confounding. Only four already available inputs, no raw-data or exposure changes.'),
('V261','Shrunk nonlinear inner-OOF calibration','inner_isotonic',
 'Original6k correction rows:3fold stratified inner splits seed52027+outerfold, fit same head on4000 and predict heldout2000. Fit nondecreasing isotonic q on6000 innerOOF probabilities/labels, clipping outside observed support and output[1e-5,1-1e-5]. Apply q to outer/test full6k-head score. p_corrected=.95*p_V25_corrected+.05*q(p_V25_corrected); final75:25originalSpline77.',
 'V186 tested affine and V189 group calibration. This tests nonlinear common score calibration without group prevalence shifts like ALL66. Positive95% anchor preserves strict corrected-Kumo ordering; standalone KS cannot improve from this transform, but fixed fusion with spline can change ranking. Inner4k/full6k distribution mismatch and isotonic overfit remain; assess only outer results.')]
rows=[dict(common,id=i,label=l,method=m,formula=f,rationale=r,strength=.05,
           families=['existing_latent','existing_coverage'],
           penalty={'posterior_quadrature':0.,'source_jackknife':.003,'function_space_ridge':.003,'monotone_credit_expert':.01,'inner_isotonic':0.}[m],
           effect_variance_penalty=.01 if m=='function_space_ridge' else 0.) for i,l,m,f,r in items]
(V/'conservative_mechanisms_plan.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
p=V/'conservative_mechanisms/reports';p.mkdir(parents=True,exist_ok=True)
with (p/'FEATURE_CATALOG.csv').open('w',newline='',encoding='utf-8') as out:
    w=csv.DictWriter(out,fieldnames=['experiment','feature_name','formula','source','cutoff','validation_status']);w.writeheader()
    for row in rows:w.writerow(dict(experiment=row['id'],feature_name=row['method'],formula=row['formula'],source='Existing V25 latent/coverage columns and frozen scores',cutoff='Inherit V25 information cutoff; all new learned transformations use outer-training only',validation_status='UNVALIDATED'))
doc=['# Five conservative mechanism experiments V257-V261','',
 'V25 remains public incumbent0.56822. Kumo100%frozen, same5outerfolds. Each experiment has one fixed5% contribution; no search over contribution strength. Existing baseline provenance uncertainties remain.','',
 'Evidence: V251KS0.53011664 vsV25KS0.53028466; deltaAUC-0.000047904, bootstrap95%deltaAUC[-0.000094094,-0.000003109]. Its top10% capture improved5 cases versusV25 while top32.37% lost8; a local capture gain is not a global improvement. Earlier item-family ablations also lowered AUC. ALL66 public0.55983 is not evidence that reducing backbone epochs helps: those backbones were frozen.','',
 'Peer review: source joint_model.py uses mean/SD tensor interactions, not posterior quadrature. reverse_synthetic_patterns.py confirms posterior SD on price-log and bureau-score scales. Original global reference estimates were not rebuilt and are not newly certified leakage-safe.','',
 'Training-label use: V257 none; V258/V259 only original6k; V260 outer56k; V261 inner3fold heads and calibration inside6k. All use identical untouched-by-this-fit outer14k, but historical reuse means these are development folds, not an independent final evaluation.','',
 'Per result: paired bootstrap KS/AUC; fold consistency; errors/top-risk capture/history groups; parameter count and training gap; source/checkpoint hashes; OOF privately; submission CSV; aggregate master/GitHub update. No automatic Kaggle submission.']
for row in rows:doc+=['','## '+row['id']+' '+row['label'],'',row['rationale'],'',row['formula']]
(V/'CONSERVATIVE_MECHANISMS_PLAN.md').write_text('\n'.join(doc)+'\n',encoding='utf-8')
print('DEFINED', [r['id'] for r in rows])

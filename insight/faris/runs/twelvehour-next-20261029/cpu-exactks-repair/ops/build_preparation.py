"""Root preparation only. No clock, fit, allocation or score selection."""
from pathlib import Path
import hashlib,json,shutil,datetime
R=Path(__file__).resolve().parents[1];P=R.parent/'twelvehour-20261029'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(v,p):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
designs={}
cpu_views=['global48','global99','actuarial99','core99','global128','global256','augmented256','global512']
for view in cpu_views:
 for mode in ['glm','intercept','glm_only']:
  for C in ([.01,.1]if mode!='intercept'else[.1]):
   for leaves in ([7,31]if mode!='glm_only'else[7]):
    key=f'cann_{view}_{mode}_c{str(C).replace(".","p")}_l{leaves}'
    designs[key]=dict(family='cann_tree',resource='cpu',view=view,admission_group=key,max_seconds=1200,offset_mode=mode,glm_C=C,num_leaves=leaves,parameters={},hypothesis='Fixed BernoulliGLMlogit offset plus nonlinear tree correction, matched intercept-only tree and GLM-only controls; each offset and correction fits only permitted base rows. Transfer of CANN principle, not exact neural CANN. Full encoder fit42500inner only; GLM/refit50000.',sources=['https://doi.org/10.1017/asb.2018.42','https://www.mdpi.com/1999-4893/16/2/99','https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Dataset.html'])
gpu_views=['global48','global64','global80','global99','actuarial99','core99']
for size in ['medium','large']:
 for view in gpu_views:
  for ratio in [1,3,5]:
   for estimators in [2,8]:
    for sampling in ['case_control','natural_matched_count']:
     key=f'cc_{size}_{view}_r{ratio}_e{estimators}_{sampling}'
     designs[key]=dict(family='kumo_casecontrol',resource='gpu',view=view,admission_group=key,max_seconds=1800,size=size,negative_per_positive=ratio,estimators=estimators,context_sampling=sampling,engine='concat_context_vocab',prediction_batch=256,memory_fraction=.7,prior_correction='none',aggregation='single_context',parameters={},hypothesis='Retain all supplied context default positives and sample1/3/5unique negatives per positive, versus natural-prevalence stratified subset of exact same derived N; 2/8estimators assess interaction with internal ensembling. No query-dependent sampling. Analogy from case-control sampling; no Kumo invariance or calibrated-risk claim.',sources=['https://arxiv.org/abs/1306.3706','https://www.stat.berkeley.edu/~wfithian/papers/lcc.pdf','https://huggingface.co/nvidia/Kumo-Tabular'])
# Paired adjacent designs stay together on one backend/GPU type.
owners={}
cpu=list(k for k in designs if designs[k]['resource']=='cpu')
for name,slots in [('cpu2001',4),('cpu2003',4),('cpu119',6)]:owners[name]=dict(resource='cpu',slots=slots,designs=[],gpu_indices=[])
for i,view in enumerate(cpu_views):owners[['cpu2001','cpu2003','cpu119'][i%3]]['designs'] +=[k for k in cpu if designs[k]['view']==view]
for n in range(1,12):owners[str(n)]=dict(resource='gpu',slots=1,designs=[],gpu_indices=[0],requested_gpu='A100'if n==2 else'T4')
owners['gpu2003']=dict(resource='gpu',slots=2,designs=[],gpu_indices=[0,1])
for size,names in [('medium',[str(n)for n in range(1,12)if n!=2]),('large',['2','gpu2003','gpu2003'])]:
 pairs=[]
 # First lower-cost ratios/e2 across allviews, then estimator interaction, then ratio5.
 for ratio,estimators in [(1,2),(3,2),(5,2),(1,8),(3,8),(5,8)]:
  for view in gpu_views:pairs.append([k for k,v in designs.items()if v.get('size')==size and v['view']==view and v['negative_per_positive']==ratio and v['estimators']==estimators])
 for i,pair in enumerate(pairs):assert len(pair)==2;owners[names[i%len(names)]]['designs']+=pair
for key,value in owners.items():
 value['planned_fits']=len(value['designs'])*10
 value['pilot_tasks']=[f'{d}__s2026__f0'for d in value['designs']]
assert len(designs)==208 and sum(v['planned_fits']for v in owners.values())==2080
cfg=dict(campaign='twelvehour-next-20261029',designs=designs,seeds=[2026,7731],folds=5,hours=12,admission_hours=10.5,model_hours=11.25,admission_safety_factor=1.5,admission_finish_margin=300,group_pilot_max_seconds=1800,pilot_tasks=[f'{k}__s2026__f0'for k in designs],group_pilots={k:[f'{k}__s2026__f0']for k in designs},priority_groups=list(designs),no_new_features=True,no_refit=True,no_submission=True)
dump(cfg,R/'CONFIG.json');dump(dict(owners=owners,whole_design_ownership=True,all11profiles_checked=True,refill='continuously while healthy; explicit30minute resource/health reconciliation; only preregistered unattempted cohorts and originalclock',source_scores_used_for_admission=False,unit_prefix='logika-next-20261029'),R/'EXECUTION_PLAN.json')
contract=json.loads((P/'VALIDATION_CONTRACT.json').read_text());contract.update(protocol_id='logika-v3-cann-casecontrol-fixed208-20261029',previous_winner=.5312163876106586,execution_limits='12h fresh immutable clock; admission10.5h, models11.25h, all12h. Runtime/finite/replay pilot perdesign before9remainingfits, safetyfactor1.5/300sec finishmargin, automatic cohort refill, no implicit retries.',scope='Supplied competition labels only, pinned foundation checkpoints; lab2001/2003 actualhome, host119 actualhome,11explicit Colab profiles. No node2002, CPUKumo, fittedweights/KV, refit, submissions or leaderboard probes.',kumo_context_policy='All context positives retained in casecontrolarms with1/3/5negatives perpositive vs natural prevalence exactN controls;2/8internalestimators. Infeasible ratios fail explicitly. Query rows/labels do not guide sampling.',composition='Identical four prespecified modes standalone/fixed.10/heldmetaadaptivegrid/oldpenalty.03stack; no outer labels for basefit,stopping,composition. Allhistorical exposure persists.')
contract['immutable_input_metadata']={name:sha(R/name)for name in ['INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','evidence/rules.md']}
dump(contract,R/'VALIDATION_CONTRACT.json')
dump(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),human_budget_hours=12,previous6hourpreparation_superseded_before_clock=True,registered_designs=208,planned_fits=2080,planned_complete_cohort_compositions=832,all70k_reused_development=True,independent_audit=False,no_fittedweights_or_KVcaches=True,feature_bank=2129,feature_selection_regeneration=False,CPU_designs=64,GPU_designs=144,all_profiles_explicit=True,public_incumbent_V25=.56822),R/'PLAN.json')
(R/'README.md').write_text('''# Twelve-hour CANN and case-control campaign

Human requested a fresh twelve-hour hard ceiling, all11Colab profiles and available lab2001/2003/host119capacity, with30minute checks and laptop-independent execution. Earlier completed studies and clocks stay sealed. This campaign preregisters208distinct designs/2080planned fits,64base-level GLM-offset tree/control recipes and144matched context-sampling foundation-model recipes. Planned counts are not completed counts.

Features and five fold input packets remain unchanged. Every design has2seeds×5folds. Runtime/finite/provenance/replay qualification runs onefullsize pilot, then admits the remaining9fits only if conservative whole-cohort completion fits the original clock. Healthy slots take the next preregistered cohort immediately. All successes, partials, execution failures and unattempted resource-limited designs remain recorded. No automatic failed-fit retries or recipe changes. No fitted weights or KV dumps.

Use immutable CLOCK.json after launch. Admissions end10.5hours afterstart, models end11.25hours, allcompute and owned billing end12hours. Final45minutes reserve collection, frozen selection, paired2000-replicate conditional development intervals and independent artifact/score replay. All70000cases remain adaptively reused development; no untouched audit exists. Baseline0.5306361042, currentdevelopmentpoint0.5312163876 and publicV250.56822 are separate. No newrefit, Kaggle submissions or leaderboard probing in this training loop.

Canonical node2001 holds scientific artifacts; isolated owner roots avoid collisions. Persistent host119 owns Colab creation/control/collection/release, eachVM also has an independent kernel deadline release thread. Never touch unrelated sessions/workloads or node2002. Git publication is sanitized; private compact evidence is backed up to the user's verified helmaDrive folder and a second remote host. Drive archival excludes credentials, virtualenvs, pretrained downloads and fittedweights. Actual running/release/backup state is in LIVE_STATE, per-owner collection receipts and Drive backup receipts.
''')
dump({str(p.relative_to(R)):sha(p)for p in sorted((R/'source').glob('*.py'))},R/'SOURCE_SEAL.json')
print(json.dumps(dict(designs=len(designs),planned_fits=2080,owners={k:len(v['designs'])for k,v in owners.items()},source_seal_sha256=sha(R/'SOURCE_SEAL.json'))))

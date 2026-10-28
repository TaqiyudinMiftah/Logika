"""Read existing sources/manifests and recover the seven selected backbone recipes."""
import ast,hashlib,json
from pathlib import Path
import pyarrow.parquet as pq

EXP=Path('/mnt/home/faris/logika-faris-exploration');FRZ=Path('/mnt/home/faris/logika-frozen-shadow-20261026');JOINT=Path('/mnt/home/faris/logika-joint-20261028');OUT=Path('/mnt/home/faris/logika-antigravity-refit-20261028')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
source=JOINT/'build_unified_tables_and_run_kumo.py';tree=ast.parse(source.read_text());names={'c90','redundant_9','c81_core','popA_active_core','popA_add','view_popA','popBC1_add','view_popBC1','popC2_add','view_popC2'}
selected=[]
for node in tree.body:
    if isinstance(node,ast.FunctionDef)and node.name=='check_cols':selected.append(node)
    elif isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in names for t in node.targets):selected.append(node)
env={'Path':Path,'json':json,'FRZ_ROOT':FRZ,'avail_set':set(pq.read_schema(JOINT/'fold0/expanded.parquet').names)}
exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),env)
views=read(FRZ/'FEATURE_FREEZE.json')['views']
models={
 'fe_cov':{'engine':'legacy_separate','columns':read(EXP/'coverage-information-fe-20261021/columns.json'),'seed_base':2026,'source':str(EXP/'coverage-information-fe-20261021/source/worker.py')},
 'fe_mat':{'engine':'legacy_separate','columns':read(EXP/'mature-mask-preserved-fe-20261021/columns.json'),'seed_base':2026,'source':str(EXP/'mature-mask-preserved-fe-20261021/source/worker.py')},
 'tr_mat_orig':{'engine':'legacy_separate','columns':read(EXP/'kumo-coverage-placement-20261021/conditions/fold0/control90/boundary.json')['feature_columns'],'seed_base':2026,'source':str(EXP/'transfer-20261025/source/core_maturity.py'),'Kumo_replacements':{'v3_repay_all_late_1_posterior_24':'matched_beta_late_1','v3_repay_all_late_30_posterior_24':'matched_beta_late_30'}},
 'kumo56':{'engine':'legacy_separate','columns':read(EXP/'nested-correction-20261021/columns.json'),'seed_base':2026,'source':str(EXP/'nested-correction-20261021/source/worker.py'),'historical_correction_raw_source':'five44800-context inner-OOF predictions perouterfold','historical_validation_raw_source':'56000context final fit'},
 'champ_s7731':{'engine':'concat_context_vocab','columns':views['champion90'],'seed_base':7731,'source':str(FRZ/'source/model_worker.py')},
 'shadow91':{'engine':'concat_context_vocab','columns':views['shadow91'],'seed_base':2026,'source':str(FRZ/'source/model_worker.py')},
 'pop_spec':{'engine':'concat_context_vocab','specialists':{'A':{'columns':env['view_popA'],'seed_base':31028},'BC1':{'columns':env['view_popBC1'],'seed_base':31029,'augmentation':'append5044C2context rows with lowest repayment_installment_count, stablemergesort'},'C2':{'columns':env['view_popC2'],'seed_base':31030}},'fold_seed_increment':10,'source':str(source)},
}
for k,v in models.items():
 v.update(size='large',num_estimators=8,estimator_batch_size=1,prediction_batch=512,precision='CUDAfloat16AMP',generator='torch.Generator(cuda)',source_sha256=sha(v['source']))
 if k!='pop_spec':v['fold_seed_increment']=1
 v['feature_widths']={part:len(s['columns'])for part,s in v['specialists'].items()}if k=='pop_spec'else len(v['columns'])
 assert all(n<=99 for n in v['feature_widths'].values())if isinstance(v['feature_widths'],dict)else v['feature_widths']<=99
out={'status':'RECOVERED_FROM_EXISTING_SOURCE_NOT_A_RUNTIME_REPLAY_GATE','models':models,'engine_differences':{'legacy_separate':'context and query TableTensor encoded separately, stringcategoricals, numeric +/-inf replacedNaN; memfraction.90','concat_context_vocab':'context/query concatenated before TableTensor construction, categorical vocabulary restricted tocontext; numeric infunchanged; memfraction.85'},'fitted_weights_or_KV_serialized':False}
(OUT/'ENGINE_CONFIGS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v['feature_widths']for k,v in models.items()}))

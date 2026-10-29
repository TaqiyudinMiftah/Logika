from pathlib import Path
import json,hashlib,itertools,shutil,py_compile
P=Path('a100/faris/overnight-refit-20261030');F=P/'book-feature-cann';R=P/'gpu-book-features';R.mkdir(exist_ok=True)
for n in ['source','ops','evidence']:shutil.copytree(F/n,R/n,dirs_exist_ok=True)
for n in ['INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','VALIDATION_CONTRACT.json','CONFIG.json','EXECUTION_PLAN.json','AGENTS.md','PARENT_CLOCK_REFERENCE.json']:(R/n).write_bytes((F/n).read_bytes())
(R/'source/context_recipe.py').write_bytes((P/'source/context_recipe.py').read_bytes())
c=json.loads((F/'CONFIG.json').read_text());designs={}
for view,mode,size,estimators,context in itertools.product(['actuarial99','global99'],['asinh','rank_normal'],['large','medium'],[2,8],[25000,50000]):
 name=f'gpu_book_{view}_{mode}_{size}_e{estimators}_n{context}'
 designs[name]=dict(admission_group=name,view=view,book_transform=mode,size=size,estimators=estimators,context_count=context,max_seconds=900,parameters={},family='BookFeature_Kumo',resource='gpu',hypothesis='Context-only tail/quantile representations can change pretrained numerical token geometry while holding feature information and context count fixed.',sources=['user:LDAEd2.pdf#PDFpage51','user:Credit-Scoring--Theory,-Methods,-and-Practice.pdf#PDFpage389'])
c.update(campaign='gpu-book-features',designs=designs,primary=next(iter(designs)),pilot_tasks=[f'{d}__s2026__f0'for d in designs],priority_designs=list(designs),expected_fits=len(designs)*10,notes='64 configurations/640planned GPU fits; runtime gates may leave many unattempted. Fresh book-derived99-column representations, same old controls. Research stops22:00UTC to reserve production refit.')
for k,v in list(c.items()):
 if isinstance(v,list)and any(isinstance(x,str)and x.startswith('book_')for x in v):c[k]=list(designs)
(R/'CONFIG.json').write_text(json.dumps(c,indent=2)+'\n')
plan={'owners':{'gpu2003':dict(designs=list(designs),pilot_tasks=c['pilot_tasks'],planned_fits=len(designs)*10,resource='gpu',slots=2,gpu_indices=[0,1])}};(R/'EXECUTION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');(R/'EXTRA_DESIGNS.jsonl').write_text('')
contract=json.loads((F/'VALIDATION_CONTRACT.json').read_text());contract.update(protocol_id='logika-v6-book-feature-Kumo-fixed64',CPU_protocol_change='Not applicable: same paired research protocol; newGPU numerical representations only.',feature_policy='Frozen99input identity with numeric asinh/quantile replacement fit42500inner only; no new rawfields. Original snapshots unchanged.')
(R/'VALIDATION_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
s=(F/'ops/focal_coordinator.py').read_text();start=s.index('SPECS=');end=s.index('\nFILES=',start)
s=s[:start]+"SPECS={'gpu2003':dict(ssh=['ssh','-o','BatchMode=yes','-p2003','faris@gpu-labkc.ub.ac.id'],root='/mnt/home/faris/logika-gpubook0630-20261030/owner2003',python=TORCH3,packets='/mnt/home/faris/logika-twelvehour-20261029/packets',memory='24G')}"+s[end:]
s=s.replace('focal0630','gpubook0630').replace('ClassWeighted_Focal_CANN','BookFeature_Kumo').replace("derived_budget_only=True)","derived_budget_only=True)\n clock['admission_end_epoch']=min(clock['admission_end_epoch'],1793308500.0);clock['model_deadline_epoch']=min(clock['model_deadline_epoch'],1793311200.0)")
(R/'ops/gpu_book_coordinator.py').write_text(s)
(R/'README.md').write_text('''# Book feature maps with Kumo
64 configurations/640 planned fits, not completion: two99-column views × two numeric maps × Large/Medium ×2/8estimators ×25k/50kcontexts. Fit numeric transforms42500inner only. New representation replaces originalnumeric values rather than exceeding99inputs; categorical fields preserved. Matched rawcontrols already archived, not rerun. Purpose: test whether pretrained numeric-token geometry benefits from actuarial tail compression, beyond earlier PLE/periodic CANN embeddings. Two idle node2003 GPUs, gated fullsize pilots then remaining9fits. Queues wait GPUidle and cannot disturb current full-data refits/foreignwork. New separateclock inherits23:30UTC totalceiling but research admission21:15/model22:00UTC, reserves fresh productionfit. No old clock edited. All70k adaptive development; no independentconfirmation. Exact benchmark gates/tolerances preserved. If fully audited candidate winslocalKS, freshfullrefit/CSV audit required before deadline upload. No V25fallback.
''')
print('Prepared',len(designs),'GPU book designs')

from pathlib import Path
import json,itertools,hashlib,shutil,py_compile
P=Path('a100/faris/overnight-refit-20261030');F=P/'neural-focal-cann';R=P/'book-feature-cann';R.mkdir(exist_ok=True)
for n in ['source','ops','evidence']:shutil.copytree(F/n,R/n,dirs_exist_ok=True)
for n in ['INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','VALIDATION_CONTRACT.json','CONFIG.json','EXECUTION_PLAN.json','AGENTS.md','PARENT_CLOCK_REFERENCE.json']:(R/n).write_bytes((F/n).read_bytes())
s=Path('a100/faris/twelvehour-next-20261029/neural-cann/source/cpu_models.py').read_text().replace('from cpu_offset import FrozenInput,_glm,_ids,_hash,_ks,_check','from cpu_offset import _glm,_ids,_hash,_ks,_check\nfrom book_features import BookInput as FrozenInput').replace('encoder=FrozenInput().fit(frame,ctx[inner])',"encoder=FrozenInput(mode=spec['book_transform']).fit(frame,ctx[inner])")
s=s.replace("model_family='Bernoulli_neural_CANN'", "model_family='BookFeature_Bernoulli_CANN',book_transform=spec['book_transform']")
(R/'source/cpu_models.py').write_text(s)
config=json.loads((F/'CONFIG.json').read_text());designs={}
for view,offset,width,residual,transform in itertools.product(['actuarial99','global99'],['intercept','glm'],[64,128],[0.,.1],['asinh','rank_normal','both']):
 name=f'book_{view}_{offset}_w{width}_r{str(residual).replace(".","p")}_{transform}'
 designs[name]=dict(admission_group=name,view=view,offset_mode=offset,width=width,glm_C=.1,residual_penalty=residual,learning_rate=.0003,book_transform=transform,max_seconds=900,parameters={},family='BookFeature_CANN',resource='cpu',hypothesis='Heavy-tailed financial ratios may be compressed by original mean/SD plus10SD clipping; append training-only robust signed-asinh and/or empirical-normal quantile features. Compare exact raw-feature CANN controls.',sources=['user:LDAEd2.pdf#PDFpage51','user:Credit-Scoring--Theory,-Methods,-and-Practice.pdf#PDFpage389','user:978-3-031-12409-9.pdf#section7.5.1'])
config.update(campaign='book-feature-cann',designs=designs,primary=next(iter(designs)),pilot_tasks=[f'{d}__s2026__f0'for d in designs],expected_fits=len(designs)*10,priority_designs=list(designs),notes='48 prospectively defined feature-engineering CANN controls;480 planned fits. Exact old raw CANN controls reused.')
for k,v in list(config.items()):
 if isinstance(v,list)and any(isinstance(x,str)and x.startswith('focal_')for x in v):config[k]=list(designs)
(R/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
plan=json.loads((F/'EXECUTION_PLAN.json').read_text());owners=list(plan['owners'])
# Whole-design balanced allocation proportional to CPU slots.
cycle=sum(([o]*plan['owners'][o]['slots']for o in owners),[])
for owner in owners:
 names=[d for j,d in enumerate(designs)if cycle[j%len(cycle)]==owner];plan['owners'][owner].update(designs=names,pilot_tasks=[f'{d}__s2026__f0'for d in names],planned_fits=len(names)*10)
(R/'EXECUTION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');(R/'EXTRA_DESIGNS.jsonl').write_text('')
contract=json.loads((F/'VALIDATION_CONTRACT.json').read_text());contract.update(protocol_id='logika-v5-book-feature-CANN-fixed48',CPU_protocol_change='Append training-only asinh/empirical-normal numerical features to original representation; no new raw records or labels.',feature_policy='Source-distinct reopened feature study. Original frozen bank and lists unchanged. New derived mappings fit42500inner covariates only, retain during50k refit.',early_stopping='Original KS stopping120epochs/patience12.',execution_limits='Same overnight epoch ceiling; owner waits old focal owner BACKEND_FINAL before starting.')
(R/'VALIDATION_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
s=(F/'ops/focal_coordinator.py').read_text().replace('focal0630','book0630').replace('ClassWeighted_Focal_CANN','BookFeature_CANN')
s=s.replace("if not(R/f'DISPATCH_{owner}.json').exists():setup(owner)","if not(R/f'DISPATCH_{owner}.json').exists() and (P/'neural-focal-cann/private/owner_metadata'/owner/'BACKEND_FINAL.json').exists():setup(owner)")
s=s.replace("errors={}\n  for owner in SPECS:","errors={}\n  for owner in SPECS:\n   if not(R/f'DISPATCH_{owner}.json').exists() and (P/'neural-focal-cann/private/owner_metadata'/owner/'BACKEND_FINAL.json').exists():setup(owner)\n  for owner in SPECS:\n   if not(R/f'DISPATCH_{owner}.json').exists():continue")
(R/'ops/book_coordinator.py').write_text(s)
(R/'README.md').write_text('''# Book-derived feature engineering
48 designs/480 planned fits. Append asinh(value / inner median-positive-absolute-scale), empirical-normal quantile maps, or both, to raw CANN inputs. Two existing99-column views, two offsets, two widths and two residual penalties; exact raw controls reused. New predictor columns are real feature maps learned from42500inner covariates only, never stopper/meta/outer labels. All prior banks and seals remain unchanged. Missingness persists. The asinh signed extension and quantile map are proposed adaptations, not formulas claimed verbatim from the books. Loss Data Analytics PDFp51 motivates tail compression; Credit Scoring PDFp389 discusses log/Winsor transforms of financial ratios; Wuthrich/Merz section7.5.1 discusses scaling. Orth's survival thesis requires lifetime/censoring labels that this default task does not supply, so no fabricated survival target is admitted. All70k reused adaptive development, no untouched audit. Starts each owner's queue only after its focal CANN queue ends. Same06:30WIB deadline, no extension. Best complete audited local candidate is full-refitted, audited and eligible for the deadline submission; no V25 fallback.
''')
print('Prepared',len(designs),'book feature designs')

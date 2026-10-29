from pathlib import Path
import json,hashlib,itertools,shutil,py_compile
P=Path('a100/faris/overnight-refit-20261030');F=P/'gpu-book-features';R=P/'helma-book-context10k';R.mkdir(exist_ok=True)
for n in ['source','ops','evidence']:shutil.copytree(F/n,R/n,dirs_exist_ok=True)
for n in ['INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','VALIDATION_CONTRACT.json','CONFIG.json','AGENTS.md','PARENT_CLOCK_REFERENCE.json']:(R/n).write_bytes((F/n).read_bytes())
s=(F/'source/gpu_models.py').read_text()
s=s.replace("encoder=BookInput(mode=spec['book_transform']).fit(frame,ctx[inner]);recipe=encoder.recipe();transformed=frame.copy();mapped=encoder._maps(encoder._numeric(frame,np.arange(len(frame))))\n for j,c in enumerate(encoder.numeric):transformed[c]=mapped[:,j].astype('float32')", "recipe={'mode':'raw_control','no_new_preprocessing':True};transformed=frame.copy()\n if spec['book_transform']!='raw':\n  encoder=BookInput(mode=spec['book_transform']).fit(frame,ctx[inner]);recipe=encoder.recipe();mapped=encoder._maps(encoder._numeric(frame,np.arange(len(frame))))\n  for j,c in enumerate(encoder.numeric):transformed[c]=mapped[:,j].astype('float32')")
s=s.replace("checkpoint=load('ENVIRONMENT.json')['checkpoint_'+size]", "runtime=load('ENVIRONMENT_LIVE.json') if (R/'ENVIRONMENT_LIVE.json').exists() else load('ENVIRONMENT.json');checkpoint=runtime['checkpoint_'+size]")
(R/'source/gpu_models.py').write_text(s)
c=json.loads((F/'CONFIG.json').read_text());designs={}
for view,mode,size,estimators in itertools.product(['global99','actuarial99'],['raw','asinh','rank_normal'],['large','medium'],[2,8]):
 name=f'helma_book10k_{view}_{mode}_{size}_e{estimators}'
 designs[name]=dict(admission_group=name,view=view,book_transform=mode,size=size,estimators=estimators,context_count=10000,max_seconds=600,parameters={},family='BookFeature_Kumo',resource='gpu',hypothesis='Test interaction of inner-only numeric tail/quantile transforms with a small natural10k context; exact same-session raw controls distinguish representation from context size. Prior25k/50k transformed context studies differ; raw10k controls are explicit replications.',sources=['user:LDAEd2.pdf#PDFpage51','user:Credit-Scoring--Theory,-Methods,-and-Practice.pdf#PDFpage389'])
c.update(campaign='helma-book-context10k',designs=designs,primary=next(iter(designs)),pilot_tasks=[f'{d}__s2026__f0'for d in designs],priority_designs=list(designs),expected_fits=len(designs)*10,notes='24designs/240plannedfits, eight raw controls and16new transform/context interactions. A100account2 explicitly authorized; no success count implied; no paid runtime after original23:30UTC ceiling.')
for k,v in list(c.items()):
 if isinstance(v,list) and any(isinstance(x,str) and x.startswith('gpu_book_')for x in v):c[k]=list(designs)
(R/'CONFIG.json').write_text(json.dumps(c,indent=2)+'\n');(R/'EXTRA_DESIGNS.jsonl').write_text('')
plan={'owners':{'2':dict(designs=list(designs),pilot_tasks=c['pilot_tasks'],planned_fits=len(designs)*10,resource='gpu',slots=1,gpu_indices=[0])}}
(R/'EXECUTION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
contract=json.loads((F/'VALIDATION_CONTRACT.json').read_text());contract.update(protocol_id='logika-v8-helma-book10k24',feature_policy='Same frozen99fields; optional asinh/quantile replacement fit42500inner only. Exact raw controls. Context10000 natural stratified draw; no query labels. No prior study source or clock modified.')
(R/'VALIDATION_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
(R/'README.md').write_text('''# Helma book-transform and small-context interaction
24designs/240plannedfits: two99-column views × raw/asinh/empirical-normal numeric map × Large/Medium ×2/8estimators, single10k natural context. Eight raw designs are explicitly repeated controls, not novel methods;16 representation/context combinations are distinct from current25k/50k transform study. Same deterministic contextdraws,42500inner-only maps and50000permitted training labels;6kmeta/14kouter labels excluded from base fit. Old70k validation is adaptively reused, no untouched audit. Exact SciPyKS, two seeds/fivefold complete cohorts, four original compositions and original numerical replay thresholds. Fullsize pilot gates each design with1.5safety/300sec margin; no automatic retry. Helmaaccount2 only, current human explicit authorization. Research admissions21:15UTC, models22:00UTC, total23:30UTC inherited unchanged. Runtime physical checkpoints stay/root; pinned identities unchanged. Every30sec compact evidence collection; owned release independent of laptop. Winner may replace currentCSV only after complete audit, higher local point and fresh70krefit. User accepts exploratory submission; no public/private gain claim. Keep every outcome/source/params and private predictions, no fitted weights or KV caches.
''')
for p in (R/'source').glob('*.py'):py_compile.compile(str(p),doraise=True)
(R/'SOURCE_SEAL.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((R/'source').rglob('*.py'))},indent=2)+'\n')
print(json.dumps({'designs':len(designs),'planned':len(designs)*10,'raw_controls':8,'new_interactions':16}))

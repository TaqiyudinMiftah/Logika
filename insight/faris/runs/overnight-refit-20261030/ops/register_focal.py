"""Prepare a source-distinct focal CANN study; no training before publication receipt."""
from pathlib import Path
import itertools,json,hashlib,shutil,subprocess,py_compile
P=Path('a100/faris/overnight-refit-20261030');R=P/'neural-focal-cann';OLD=Path('a100/faris/twelvehour-next-20261029/neural-softks-cann');R.mkdir(exist_ok=True)
for n in ['source','ops','evidence']:shutil.copytree(OLD/n,R/n,dirs_exist_ok=True)
for n in ['INPUTS.json','ENVIRONMENT.json','CHECKPOINTS.json','VALIDATION_CONTRACT.json','CONFIG.json','EXECUTION_PLAN.json','AGENTS.md']:(R/n).write_bytes((OLD/n).read_bytes())
s=Path('a100/faris/twelvehour-next-20261029/neural-cann/source/cpu_models.py').read_text()
s=s.replace("def epochs(net,x,y,offset,spec,seed,end,number,stop=None):",'''def focal(logits,y,gamma,mode):
    pi=y.mean()
    # Caller passes full training prevalence, not minibatch prevalence.
    pos=mode
    probability=torch.sigmoid(logits);pt=torch.where(y>0,probability,1-probability)
    weight=torch.where(y>0,torch.as_tensor(pos,dtype=logits.dtype),torch.ones_like(y))
    return torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*weight*(1-pt).pow(gamma)

def epochs(net,x,y,offset,spec,seed,end,number,stop=None):''')
s=s.replace("penalty=float(spec['residual_penalty']);best_epoch", "prevalence=float(y.mean());positive_weight=(1-prevalence)/prevalence if spec['class_mode']=='balanced' else 1.;normalizer=prevalence*positive_weight+1-prevalence\n    penalty=float(spec['residual_penalty']);best_epoch")
s=s.replace("torch.nn.functional.binary_cross_entropy_with_logits(offset[index]+residual,y[index])", "focal(offset[index]+residual,y[index],float(spec['focal_gamma']),positive_weight).mean()/normalizer")
s=s.replace("model_family='Bernoulli_neural_CANN'", "model_family='ClassWeighted_Focal_Bernoulli_CANN',focal_gamma=float(spec['focal_gamma']),class_mode=spec['class_mode']")
(R/'source/cpu_models.py').write_text(s)
config=json.loads((OLD/'CONFIG.json').read_text());designs={}
for view,offset,width,residual,mode,gamma in itertools.product(['actuarial99','global99'],['intercept','glm'],[64,128],[0.,.1],['natural','balanced'],[0.,1.,2.]):
 if mode=='natural'and gamma==0:continue
 name=f'focal_{view}_{offset}_w{width}_r{str(residual).replace(".","p")}_{mode}_g{int(gamma)}'
 designs[name]=dict(admission_group=name,view=view,offset_mode=offset,width=width,glm_C=.1,residual_penalty=residual,learning_rate=.0003,focal_gamma=gamma,class_mode=mode,max_seconds=900,parameters={},family='ClassWeighted_Focal_CANN',resource='cpu',hypothesis='Focus gradients on hard examples/class-balanced failures while retaining exact CANN controls and held-meta calibration; no assumed gain.',sources=['https://arxiv.org/abs/1708.02002v2','https://doi.org/10.1017/asb.2018.42'])
config.update(campaign='neural-focal-cann',designs=designs,primary=next(iter(designs)),pilot_tasks=[f'{d}__s2026__f0'for d in designs]);(R/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
plan=json.loads((OLD/'EXECUTION_PLAN.json').read_text());owners=list(plan['owners'])
for owner in owners:
 i=owners.index(owner);names=[d for j,d in enumerate(designs)if j%len(owners)==i];plan['owners'][owner].update(designs=names,pilot_tasks=[f'{d}__s2026__f0'for d in names],planned_fits=len(names)*10)
(R/'EXECUTION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');(R/'EXTRA_DESIGNS.jsonl').write_text('')
contract=json.loads((R/'VALIDATION_CONTRACT.json').read_text());contract.update(protocol_id='logika-v4-focal-CANN-fixed80',CPU_protocol_change='Train-only prevalence weighted BCE/focal objective; same exact neural architecture/42500 encoder/7500 stopper/fresh50000 context refit.',early_stopping='Original120epochs/patience12 and KS stopping; weight ratios fitted on permitted training rows only.',execution_limits='New explicit overnight clock, model cutoff23:00UTC, original prior12h clock untouched.')
(R/'VALIDATION_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
# Units and roots are unique, parent is the new explicit overnight campaign.
s=(OLD/'ops/softks_coordinator.py').read_text().replace('softks','focal0630').replace('SoftKS','Focal').replace('Hybrid_softKS_CANN','ClassWeighted_Focal_CANN').replace("str(P/'packets')","'/mnt/home/faris/logika-twelvehour-next-20261029/packets'")
s=s.replace("for n in ['baseline.npz','previous.npz','target.npy']:shutil.copy2(P/'evaluation_inputs'/n,R/'evaluation_inputs'/n)","for n in ['baseline.npz','previous.npz','target.npy']:shutil.copy2(Path('/mnt/home/faris/logika-twelvehour-next-20261029/evaluation_inputs')/n,R/'evaluation_inputs'/n)")
s=s.replace("if not(R/'packets').exists():(R/'packets').symlink_to(P/'packets',target_is_directory=True)","if not(R/'packets').exists():(R/'packets').symlink_to('/mnt/home/faris/logika-twelvehour-next-20261029/packets',target_is_directory=True)")
(R/'ops/focal_coordinator.py').write_text(s)
(R/'README.md').write_text('''# Prospective focal CANN adaptation
80 designs/800 planned fits; natural gamma1/2 and balanced gamma0/1/2 across two existing views, two widths, two offsets and two penalties. Exact natural-BCE controls are reused from the sealed neural CANN archive. Source is Lin et al 2017 https://arxiv.org/abs/1708.02002v2; borrowing its loss for tabular Bernoulli CANN is an adaptation, not RetinaNet reproduction. Primary abstract retrieved29October2026. Prior focal tree controls do not establish this neural result. All70k adaptive development, no untouched audit. Perdesign runtime/finite/5e-6 replay before remaining9fits. All new source/config sealed and preregistered before first fit. If a complete audited winner exceeds current ready local KS, refit on all70k with frozen columns and its exact held-meta recipe. Otherwise retain the newly trained current candidate. No historical CSV fallback. Ends under the parent06:30WIB deadline, preserving old clocks.
''')
for p in (R/'source').glob('*.py'):py_compile.compile(str(p),doraise=True)
seal={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted((R/'source').rglob('*.py'))};(R/'SOURCE_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n')
print('Prepared',len(designs),'designs')
